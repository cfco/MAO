"""MCP 接入层：连接 MCP server（stdio / streamable-http），动态发现工具并注册。

设计：
- 每个 server 一个后台线程 + 独立事件循环，保持长连接。
- 断线自动重连：连接异常后指数退避重试（1s → 2s → 4s → ... 最大 30s）。
- 工具调用通过 run_coroutine_threadsafe 提交到对应事件循环，包装成同步 Tool。
- 工具名统一加 mcp__<server>__ 前缀，避免与内置/Skill 工具冲突。
"""

from __future__ import annotations

import asyncio
import random
import re
import threading
import time
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..core.redact import safe_error_text
from .base import Tool, ToolRegistry, ToolResult

try:
    from mcp.client.streamable_http import streamablehttp_client

    _HAS_HTTP = True
except ImportError:  # 兼容旧版 mcp 包
    _HAS_HTTP = False

HTTP_TRANSPORTS = {"http", "streamable_http", "streamable-http", "streamablehttp"}

# 重连退避参数
_RECONNECT_BASE = 1.0  # 首次重连等待 1s
_RECONNECT_MAX = 30.0  # 封顶 30s
_RECONNECT_JITTER = 0.5  # 每次加 0~0.5s 随机抖动，避免多连接同时重连雪崩


def _safe_name(s: str) -> str:
    """把任意字符串归一成工具名安全片段。

    连字符 `-` 也归一为下划线：否则 server "a-b" 与 "a_b" 归一后工具名
    `mcp__a-b__x` / `mcp__a_b__x` 不重名，注册表的重名检测形同虚设，
    两个都注册成功——与"重名只留第一个、不炸 Bridge"的意图相悖（审计 P1b）。
    """
    return re.sub(r"[^A-Za-z0-9_]", "_", str(s))[:32] or "srv"


class McpTool(Tool):
    """远程 MCP 工具的本地同步包装。

    execute 必须收 (args, ctx) 两个位置参数：ToolRegistry.run 恒以
    `tool.execute(args, ctx)` 调用，只声明 args 会抛 TypeError，
    被注册表兜底吞掉后所有 mcp__* 工具永远"工具执行出错"（审计 P0）。
    """

    def __init__(
        self, name: str, description: str, input_schema: dict, conn: McpConnection, remote_name: str
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._conn = conn
        self._remote_name = remote_name

    def execute(self, args: dict, ctx: Any | None = None) -> str:
        return self._conn.call_tool(self._remote_name, args)


class McpConnection:
    """一个 MCP server 的长连接：后台线程跑事件循环，会话保持存活。

    生命周期（含重连）：
    1) start() 启动后台线程
    2) 线程内 new event loop → 循环执行 _connect_and_host()：
       - _connect_once() 建立 transport + session + initialize
       - 成功后 await 唤醒事件（host），保持会话可被 call_tool 使用
       - 唤醒（被动断开/主动 stop）后在**同一 task** 里退出 CM → 指数退避 → 重连
    3) stop() 置 _stop 并唤醒 host；收尾清理由后台线程完成

    为什么必须"同 task 进、同 task 出"（审计 P2）：stdio_client/http 的上下文管理器
    底层是 anyio cancel scope，跨 task 退出会直接抛错。旧实现 enter 在 task A、
    exit 在 host 结束后的 task B，__aexit__ 必然失败又被吞——子进程从不真正关闭。
    """

    def __init__(self, name: str, cfg: dict, call_timeout: float = 300.0):
        self.name = name
        self.cfg = cfg
        # 单次工具调用最长等待：tools.mcp_tool_timeout（原先硬编码 300，长事务会被误杀）。
        self._call_timeout = max(1.0, float(call_timeout))
        self.loop: asyncio.AbstractEventLoop | None = None
        self.session: ClientSession | None = None
        self._transport_cm: Any = None
        self._session_cm: Any = None
        self.error: str | None = None
        self._thread: threading.Thread | None = None
        # 重连状态
        self._stop = threading.Event()  # stop() 时置位，跳出重连循环
        self._reconnect_count: int = 0
        self._lock = threading.Lock()  # 保护 reconnect_count 和 session/error 的并发读写
        # host 等待事件：仅在 _connect_and_host（loop 线程）内创建/清空，
        # 其它线程通过 _wake_host()（call_soon_threadsafe 投递）唤醒它。
        self._host_evt: asyncio.Event | None = None

    # ---------- 生命周期 ----------

    def start(self) -> None:
        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_connect_forever, args=(ready,), name=f"mcp-{self.name}", daemon=True
        )
        self._thread.start()
        if not ready.wait(timeout=90):
            # error 的读写统一走 _lock（2026-11-04 审计）：后台线程连接成功时会在锁内清零
            # error，这里裸写会把"刚连上"覆盖回"超时"——调用方判失败、工具一个不注册，
            # 后台连接却活着占着 stdio 子进程。
            with self._lock:
                self.error = self.error or "连接超时（90s）"

    def _run_connect_forever(self, ready: threading.Event) -> None:
        """后台线程主循环：连接+托管 → 唤醒（断开/停止）→ 退避重连 → ... 直到 stop()。"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        ever_connected = False
        try:
            while not self._stop.is_set():
                connected = self.loop.run_until_complete(self._connect_and_host(ready))
                ready.set()  # 无论首次连接成败，都让 start() 超时逻辑能放行
                if self._stop.is_set():
                    break  # 主动关闭，不再重连
                # 首连失败直接退出（start() 拿到 error 快速上报）；曾连上过则属"断线重连"范畴
                # ——免费/不稳节点挂了又恢复是常态，必须继续退避重试。旧实现在这里 break，
                # 第一次重连失败就把线程判死刑，与模块头声明的重连承诺相悖。
                if not connected and not ever_connected:
                    break
                ever_connected = True
                self._reconnect_count += 1
                self._backoff_wait()
        except Exception as e:  # noqa: BLE001 - 线程内兜底
            self.error = self.error or f"{type(e).__name__}: {safe_error_text(e)}"
            ready.set()
        finally:
            # 线程收尾：关闭事件循环（否则每次 stop/线程退出泄漏一个 loop 及其句柄）。
            # 置 None 再 close，避免 call_tool 拿到已关闭的 loop 提交协程。
            # CM 已在 _connect_and_host 的同一 task 里退出，这里只剩关 loop。
            loop, self.loop = self.loop, None
            self.session = None
            try:
                if loop is not None:
                    loop.close()
            except Exception:  # noqa: BLE001
                pass

    async def _connect_and_host(self, ready: threading.Event) -> bool:
        """一次完整的「连接 → 托管 → 清理」：enter 与 exit 恒在同一 task 内（anyio 约束）。

        返回 True=确实建立过连接（CM 已清理）；False=连接失败（_connect_once 已回滚部分 enter）。
        host 阶段 await _host_evt：被 _wake_host 唤醒（被动断线 / 主动 stop）即收尾。
        """
        if not await self._connect_once(ready):
            return False
        # 连接建立即放行 start() 的 90s 等待——不能让 ready 等 host 结束，
        # 否则健康长连接下 start() 会一直卡到超时才返回。
        ready.set()
        ev = asyncio.Event()
        self._host_evt = ev
        try:
            if not self._stop.is_set():
                await ev.wait()
            return True
        finally:
            self._host_evt = None
            await self._shutdown_once()
            self.session = None
            self._transport_cm = None
            self._session_cm = None

    def _wake_host(self) -> None:
        """结束当前 host 等待（在 loop 线程上执行，见 stop/call_tool 的投递）。"""
        ev = self._host_evt
        if ev is not None:
            ev.set()

    async def _connect_once(self, ready: threading.Event) -> bool:
        """一次连接尝试。成功返回 True（session 建立、初始化完成）。失败返回 False。

        失败回滚：transport 已 enter 而 session 失败时必须逆序退出已进入的 CM——
        旧实现只记 error 不回滚，每次失败泄漏一个 stdio 子进程/句柄，长期重连会累积。
        """
        transport_entered = False
        session_entered = False
        try:
            transport = str(self.cfg.get("transport", "stdio")).lower()
            if transport in HTTP_TRANSPORTS:
                if not _HAS_HTTP:
                    raise RuntimeError("当前 mcp 包不支持 HTTP 传输，请升级：pip install -U mcp")
                headers = self.cfg.get("headers") or {}
                self._transport_cm = streamablehttp_client(str(self.cfg["url"]), headers=headers)
                read, write, _ = await self._transport_cm.__aenter__()
            else:
                params = StdioServerParameters(
                    command=str(self.cfg["command"]),
                    args=[str(a) for a in self.cfg.get("args", [])],
                    env=self.cfg.get("env") or None,
                )
                self._transport_cm = stdio_client(params)
                read, write = await self._transport_cm.__aenter__()
            transport_entered = True
            self._session_cm = ClientSession(read, write)
            self.session = await self._session_cm.__aenter__()
            session_entered = True
            await self.session.initialize()
            # 连接成功：清零重连计数，清 error
            with self._lock:
                self._reconnect_count = 0
                self.error = None
            return True
        except Exception as e:  # noqa: BLE001
            # 逆序回滚：只退出真正 enter 成功的部分（未 enter 的 CM 调 __aexit__ 行为未定义）
            if session_entered:
                try:
                    await self._session_cm.__aexit__(None, None, None)
                except Exception:  # noqa: BLE001 - 回滚失败不掩盖原始错误
                    pass
            if transport_entered:
                try:
                    await self._transport_cm.__aexit__(None, None, None)
                except Exception:  # noqa: BLE001
                    pass
            self.session = None
            self._session_cm = None
            self._transport_cm = None
            with self._lock:
                self.error = f"{type(e).__name__}: {safe_error_text(e)}"
            return False

    async def _shutdown_once(self) -> None:
        """退出 session / transport 上下文管理器。

        只允许由 _connect_and_host 的 finally 调用——与 __aenter__ 同一 task，
        anyio cancel scope 才能正常退出（跨 task 退出必抛错、清理形同虚设）。
        """
        try:
            if self._session_cm is not None:
                try:
                    await self._session_cm.__aexit__(None, None, None)
                except Exception:  # noqa: BLE001
                    pass
        finally:
            if self._transport_cm is not None:
                try:
                    await self._transport_cm.__aexit__(None, None, None)
                except Exception:  # noqa: BLE001
                    pass

    def _backoff_wait(self) -> None:
        """指数退避 + 随机抖动。等待期间可被 stop() 提前中断。"""
        attempt = self._reconnect_count
        delay = min(_RECONNECT_MAX, _RECONNECT_BASE * (2 ** (attempt - 1)))
        delay += random.uniform(0, _RECONNECT_JITTER)
        # 用 Event.wait 替代 time.sleep，便于 stop() 时立即中断
        self._stop.wait(timeout=delay)

    def stop(self) -> None:
        """主动关闭：置停止位并唤醒 host，收尾（退 CM → 退线程 → 关 loop）由后台线程完成。

        旧实现在这里 run_coroutine_threadsafe(_shutdown_once) + loop.stop()：
        前者跨 task 退出 anyio scope 必抛错被吞（子进程从没关成），后者会让
        run_until_complete 直接炸 RuntimeError。现在只需叫醒 host 任务，
        它会在自己（=enter 时那个）task 里把清理做完再正常返回。
        """
        self._stop.set()
        loop = self.loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(self._wake_host)
            except Exception:  # noqa: BLE001 - 线程已退出、loop 已关，无资源可漏
                pass

    # ---------- 调用 ----------

    def list_tools(self) -> list:
        if not self.session or not self.loop:
            raise RuntimeError(f"MCP server '{self.name}' 未连接：{self.error}")
        fut = asyncio.run_coroutine_threadsafe(self.session.list_tools(), self.loop)
        return list(fut.result(timeout=60).tools)

    def call_tool(self, remote_name: str, args: dict) -> str:
        with self._lock:
            session, loop = self.session, self.loop
        if not session or not loop:
            # 断线中：给出明确提示，让上层知道在等重连
            with self._lock:
                rc = self._reconnect_count
            hint = f"（已连续重连 {rc} 次，后台自动重试中）" if rc > 0 else ""
            return ToolResult(
                f"错误：MCP server '{self.name}' 未连接：{self.error}{hint}", ok=False
            )
        try:
            fut = asyncio.run_coroutine_threadsafe(session.call_tool(remote_name, args or {}), loop)
            result = fut.result(timeout=self._call_timeout)
        except Exception as e:  # noqa: BLE001
            # 超时分支必须 cancel future（2026-11-04 审计）：协程还在 loop 里持有 session，
            # 不标记取消的话它会陪到 session 作废才被吞掉异常，资源回收路径不干净。
            try:
                fut.cancel()
            except Exception:  # noqa: BLE001
                pass
            # 调用时异常（session 断了）：作废 session 并唤醒 host，走「同 task 清理→退避→重连」。
            if not self._stop.is_set():
                with self._lock:
                    self.session = None
                # 唤醒用**当场捕获的事件对象**而不是再回读 self._host_evt（2026-11-04 审计）：
                # 检查通过之后、回调执行之前 loop 线程可能已把它清空——回调里回读拿到 None
                # 直接返回，唤醒丢失，host 还挂在 ev.wait() 上永不重连（连接永久僵死）。
                ev = self._host_evt
                if ev is not None and loop.is_running():
                    try:
                        loop.call_soon_threadsafe(ev.set)
                    except Exception:  # noqa: BLE001 - loop 恰好已退出等边缘情况
                        pass
            return ToolResult(
                f"MCP 工具调用失败：{type(e).__name__}: {safe_error_text(e)}", ok=False
            )
        texts: list[str] = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                texts.append(text)
            elif type(block).__name__ == "ImageContent":
                texts.append("（返回了一张图片，本工具暂以文本为主，图片已忽略）")
        joined = "\n".join(texts)
        if getattr(result, "isError", False):
            return ToolResult("MCP 工具返回错误：" + (joined or "(无错误详情)"), ok=False)
        return joined or "(空结果)"


def connect_and_register(
    server_cfgs: list[dict],
    registry: ToolRegistry,
    out_conns: list | None = None,
    out_tools: list | None = None,
    call_timeout: float = 300.0,
) -> list[str]:
    """连接全部配置的 MCP server 并注册其工具。返回状态日志行。

    out_conns / out_tools 可选：把建立的连接与注册的工具对象一并带出，
    供进程内共享连接组复用（见 acquire_group）。
    call_timeout：单次工具调用的最长等待（tools.mcp_tool_timeout），默认 300s。
    """
    status: list[str] = []
    for cfg in server_cfgs:
        name = str(cfg.get("name") or cfg.get("command") or cfg.get("url") or "mcp")
        conn = McpConnection(name, cfg, call_timeout)
        conn.start()
        if out_conns is not None:
            out_conns.append(conn)
        if conn.error or not conn.session:
            status.append(f"[MCP] {name}: 连接失败 - {conn.error}")
            # 别留活连接空转（2026-11-04 审计）：90s 超时判失败后，后台线程若恰好在此刻
            # 连上了，会一直托管着 stdio 子进程却没有任何工具注册——stop() 收掉它，
            # 用户修好配置重启自然重连。
            try:
                conn.stop()
            except Exception:  # noqa: BLE001 - 替身连接对象可能没有 stop
                pass
            continue
        try:
            tools = conn.list_tools()
        except Exception as e:  # noqa: BLE001
            status.append(f"[MCP] {name}: 工具发现失败 - {safe_error_text(e)}")
            try:
                conn.stop()
            except Exception:  # noqa: BLE001 - 工具一个没注册，连接留着只是占资源
                pass
            continue
        prefix = _safe_name(name)
        for t in tools:
            schema = t.inputSchema if isinstance(t.inputSchema, dict) else {}
            tool = McpTool(
                name=f"mcp__{prefix}__{_safe_name(t.name)}",
                description=f"[MCP:{name}] {getattr(t, 'description', '') or t.name}",
                input_schema=schema or {"type": "object", "properties": {}},
                conn=conn,
                remote_name=t.name,
            )
            try:
                registry.register(tool)
            except ValueError as e:
                # 重名（如 server "a-b" 与 "a_b" 归一后同名）：跳过一个工具远好过
                # 炸掉整个 Bridge 初始化——后者会让 serve() 连 ready 行都发不出。
                status.append(f"[MCP] {name}: 跳过重名工具 {tool.name}（远端名 {t.name}）- {e}")
                continue
            if out_tools is not None:
                out_tools.append(tool)
        status.append(f"[MCP] {name}: 已连接，发现 {len(tools)} 个工具")
    return status


def stop_all(conns: list) -> None:
    """叫醒所有连接后台线程收尾，并等它们真的收完（上限 5s，全体共用一份预算）。

    `stop()` 只是置停止位 + 唤醒 host，退出 stdio 子进程与关 loop 都在后台线程里做。
    不 join 就返回的话，调用方（bridge.close / 组引用归零）随即走到进程退出，daemon 线程
    被硬杀 ⇒ 第三方 MCP server 的子进程留成孤儿（还占着 stdin/stdout 与端口）。
    先逐个 stop 再统一 join，多个 server 不会各自等满 5s。
    """
    for c in conns:
        try:
            c.stop()
        except Exception:  # noqa: BLE001
            pass
    deadline = time.monotonic() + 5.0
    for c in conns:
        thread = getattr(c, "_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, deadline - time.monotonic()))


# 组共享已拆到 mcp_group.py（2026-11-04，守「单文件 ≤500 行」约定）；
# 这里保留 re-export：bridge / orchestrator 的历史导入面（`from .mcp_client import
# acquire_group, register_group_tools, release_group`）逐字可用。
from .mcp_group import (  # noqa: F401,E402
    McpGroup,
    acquire_group,
    register_group_tools,
    release_group,
)
