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
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .base import Tool, ToolRegistry

try:
    from mcp.client.streamable_http import streamablehttp_client
    _HAS_HTTP = True
except ImportError:  # 兼容旧版 mcp 包
    _HAS_HTTP = False

HTTP_TRANSPORTS = {"http", "streamable_http", "streamable-http", "streamablehttp"}

# 重连退避参数
_RECONNECT_BASE = 1.0      # 首次重连等待 1s
_RECONNECT_MAX = 30.0      # 封顶 30s
_RECONNECT_JITTER = 0.5    # 每次加 0~0.5s 随机抖动，避免多连接同时重连雪崩


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(s))[:32] or "srv"


class McpTool(Tool):
    """远程 MCP 工具的本地同步包装。"""

    def __init__(self, name: str, description: str, input_schema: dict, conn: McpConnection, remote_name: str):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._conn = conn
        self._remote_name = remote_name

    def execute(self, args: dict) -> str:
        return self._conn.call_tool(self._remote_name, args)


class McpConnection:
    """一个 MCP server 的长连接：后台线程跑事件循环，会话保持存活。

    生命周期（含重连）：
    1) start() 启动后台线程
    2) 线程内 new event loop → run_connect_forever() 循环：
       - _connect_once() 建立 transport + session + initialize
       - 成功后保持 run_forever()
       - 连接断开后捕获异常 → 指数退避 → 再次 _connect_once()
    3) stop() 主动退出（设置 _stop 事件 → 线程感知后退出循环）
    """

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg
        self.loop: asyncio.AbstractEventLoop | None = None
        self.session: ClientSession | None = None
        self._transport_cm: Any = None
        self._session_cm: Any = None
        self.error: str | None = None
        self._thread: threading.Thread | None = None
        # 重连状态
        self._stop = threading.Event()      # stop() 时置位，跳出重连循环
        self._reconnect_count: int = 0
        self._lock = threading.Lock()        # 保护 reconnect_count 和 session/error 的并发读写

    # ---------- 生命周期 ----------

    def start(self) -> None:
        ready = threading.Event()
        self._thread = threading.Thread(target=self._run_connect_forever, args=(ready,), name=f"mcp-{self.name}", daemon=True)
        self._thread.start()
        if not ready.wait(timeout=90):
            self.error = self.error or "连接超时（90s）"

    def _run_connect_forever(self, ready: threading.Event) -> None:
        """后台线程主循环：连接 → 断开 → 退避重连 → ... 直到 stop()。"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        ever_connected = False
        try:
            while not self._stop.is_set():
                connected = self.loop.run_until_complete(self._connect_once(ready))
                ready.set()  # 无论首次连接成败，都让 start() 超时逻辑能放行
                if connected:
                    ever_connected = True
                    # 进入 run_forever，直到 session 因异常断开（断开时会触发 reconnect_backoff）
                    try:
                        self.loop.run_forever()
                        # run_forever 正常退出（loop.stop 被调用）：需要判断是主动 stop 还是被动断开
                    finally:
                        self._shutdown_resources()
                    if self._stop.is_set():
                        break  # 主动关闭，不再重连
                    # 被动断开：触发重连
                    self._reconnect_count += 1
                    self._backoff_wait()
                    continue
                # 连接失败：首连失败直接退出（start() 拿到 error 快速上报，语义不变）；
                # 曾连上过则说明节点活着过，此后失败属于"断线重连"范畴——免费/不稳节点
                # 挂了又恢复是常态，必须继续指数退避重试。原实现在这里 break，
                # 第一次重连失败就把重连线程判死刑，与模块头声明的重连承诺相悖。
                if not ever_connected:
                    break
                self._reconnect_count += 1
                self._backoff_wait()
        except Exception as e:  # noqa: BLE001 - 线程内兜底
            self.error = self.error or f"{type(e).__name__}: {e}"
            ready.set()
        finally:
            # 线程收尾：关闭事件循环（否则每次 stop/线程退出泄漏一个 loop 及其句柄）。
            # 置 None 再 close，避免 call_tool 拿到已关闭的 loop 提交协程。
            loop, self.loop = self.loop, None
            try:
                if loop is not None:
                    loop.close()
            except Exception:  # noqa: BLE001
                pass

    async def _connect_once(self, ready: threading.Event) -> bool:
        """一次连接尝试。成功返回 True（session 建立、初始化完成）。失败返回 False。

        失败回滚：transport 已 __aenter__ 成功后若 session 建立/初始化失败，
        必须把已进入的 CM 逆序退出。原实现只记 error 不回滚，每次失败尝试
        泄漏一个 stdio 子进程/连接句柄——重连改为长期持续后，泄漏会不断累积。
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
                self.error = f"{type(e).__name__}: {e}"
            return False

    def _shutdown_resources(self) -> None:
        """关闭 transport/session 上下文管理器（不销毁 loop，重连还要用）。

        loop 仍在跑（其它线程触发）用 run_coroutine_threadsafe 提交；
        loop 已停（本线程 run_forever 退出后收尾）必须直接 run_until_complete，
        否则提交到停住的 loop 永远不会执行，transport 句柄静默泄漏。
        """
        try:
            loop = self.loop
            if loop is None:
                pass
            elif loop.is_running():
                asyncio.run_coroutine_threadsafe(self._shutdown_once(), loop).result(timeout=5)
            else:
                loop.run_until_complete(self._shutdown_once())
        except Exception:  # noqa: BLE001 - 关闭时可能 loop 已死
            pass
        self.session = None
        self._transport_cm = None
        self._session_cm = None

    async def _shutdown_once(self) -> None:
        try:
            if self._session_cm is not None:
                await self._session_cm.__aexit__(None, None, None)
            if self._transport_cm is not None:
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
        self._stop.set()
        if self.loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._shutdown_once(), self.loop).result(timeout=5)
            except Exception:  # noqa: BLE001
                pass
            try:
                self.loop.call_soon_threadsafe(self.loop.stop)
            except Exception:  # noqa: BLE001
                pass
            # 不在此处置 self.loop = None：后台线程收尾（_run_connect_forever 的 finally）
            # 还要用它提交关闭协程并负责 close，由线程自己置空，避免竞态拿到 None 崩溃。

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
            return f"错误：MCP server '{self.name}' 未连接：{self.error}{hint}"
        try:
            fut = asyncio.run_coroutine_threadsafe(
                session.call_tool(remote_name, args or {}), loop
            )
            result = fut.result(timeout=300)
        except Exception as e:  # noqa: BLE001
            # 调用时异常（session 断了）：标记重连，让后台线程接手
            if not self._stop.is_set():
                with self._lock:
                    self.session = None  # 触发后台 run_forever 退出 → 重连
                # run_forever 只有 loop.stop() 才会退出（session=None 无人监听），
                # 必须显式停掉 loop，后台线程才会走「退避 → _connect_once」重连路径，
                # 否则该 server 从此永远返回"未连接"。
                loop = self.loop
                if loop is not None and loop.is_running():
                    try:
                        loop.call_soon_threadsafe(loop.stop)
                    except Exception:  # noqa: BLE001 - loop 恰好已退出等边缘情况
                        pass
            return f"MCP 工具调用失败：{type(e).__name__}: {e}"
        texts: list[str] = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                texts.append(text)
            elif type(block).__name__ == "ImageContent":
                texts.append("（返回了一张图片，本工具暂以文本为主，图片已忽略）")
        joined = "\n".join(texts)
        if getattr(result, "isError", False):
            return "MCP 工具返回错误：" + (joined or "(无错误详情)")
        return joined or "(空结果)"


def connect_and_register(
    server_cfgs: list[dict], registry: ToolRegistry, out_conns: list | None = None
) -> list[str]:
    """连接全部配置的 MCP server 并注册其工具。返回状态日志行。"""
    status: list[str] = []
    for cfg in server_cfgs:
        name = str(cfg.get("name") or cfg.get("command") or cfg.get("url") or "mcp")
        conn = McpConnection(name, cfg)
        conn.start()
        if out_conns is not None:
            out_conns.append(conn)
        if conn.error or not conn.session:
            status.append(f"[MCP] {name}: 连接失败 - {conn.error}")
            continue
        try:
            tools = conn.list_tools()
        except Exception as e:  # noqa: BLE001
            status.append(f"[MCP] {name}: 工具发现失败 - {e}")
            continue
        prefix = _safe_name(name)
        for t in tools:
            schema = t.inputSchema if isinstance(t.inputSchema, dict) else {}
            registry.register(McpTool(
                name=f"mcp__{prefix}__{_safe_name(t.name)}",
                description=f"[MCP:{name}] {getattr(t, 'description', '') or t.name}",
                input_schema=schema or {"type": "object", "properties": {}},
                conn=conn,
                remote_name=t.name,
            ))
        status.append(f"[MCP] {name}: 已连接，发现 {len(tools)} 个工具")
    return status


def stop_all(conns: list) -> None:
    for c in conns:
        try:
            c.stop()
        except Exception:  # noqa: BLE001
            pass
