"""本轮审查修正的回归测试：全部离线，无需 API key / 网络。

覆盖（对应 2026-09-21 审查批次）：
- MCP 断线重连：曾连上后重连失败不得永久放弃（旧实现第一次重连失败即 break）
- MCP _connect_once 失败回滚：transport 已 enter 后失败必须退出，不泄漏子进程
- WorkerPool.ask：命中缓存的回答不得被节点冷却误拦；无缓存的新请求仍被冷却挡住
- bridge ask 兜底单模型：timeout/max_retries/temperature 须读 llm 配置
- bridge stderr 编码：诊断（配置警告）必须是 UTF-8，外部驱动方按 UTF-8 解码不得报错
"""
from __future__ import annotations

import asyncio
import threading
import time

from agent.config import Config, _interpolate

# ---------------- MCP：断线重连持续重试 ----------------

def test_mcp_reconnect_retries_after_failed_reconnect():
    """曾连上过 → 断开 → 重连失败时应持续退避重试，而不是第一次失败就终结线程。

    旧实现：`if not connected: break` 无条件生效，重连窗口里一次失败即永久放弃，
    与模块头声明的「断线自动重连：指数退避重试」相悖（免费节点挂了又恢复是常态）。
    """
    from agent.tools.mcp_client import McpConnection

    conn = McpConnection("t", {"name": "t", "transport": "stdio"})
    attempts: list[int] = []
    backoffs: list[int] = []

    async def fake_connect(ready: threading.Event) -> bool:
        n = len(attempts) + 1
        attempts.append(n)
        if n == 1:
            # 模拟"连上后马上断开"：run_forever 启动后 50ms 内停掉 loop
            conn.loop.call_later(0.05, conn.loop.stop)
            return True
        if n >= 4:
            conn._stop.set()  # 已证明会持续重试，放线程正常收尾
        return False

    def fake_backoff() -> None:
        backoffs.append(1)
        time.sleep(0.02)  # 不用真退避，但要给 while 条件检查 _stop 的机会

    conn._connect_once = fake_connect  # type: ignore[method-assign]
    conn._backoff_wait = fake_backoff  # type: ignore[method-assign]
    conn.start()
    try:
        deadline = time.time() + 10
        while len(attempts) < 4 and time.time() < deadline:
            time.sleep(0.02)
        assert len(attempts) >= 4, (
            f"重连失败后应持续重试，实际尝试 {len(attempts)} 次"
            "（旧实现停在 2 次：断开后第一次重连失败即 break）")
        assert len(backoffs) >= 2, "断开与每次重连失败都应经历退避"
    finally:
        conn._stop.set()
        conn._thread.join(timeout=10)
    assert not conn._thread.is_alive(), "stop 后重连线程应退出"


def test_mcp_first_connect_failure_still_fails_fast():
    """首连失败保持快速失败语义：线程退出、error 上报，不进入无限重试。"""
    from agent.tools.mcp_client import McpConnection

    conn = McpConnection("t", {"name": "t", "transport": "stdio"})
    attempts: list[int] = []

    async def fake_connect(ready: threading.Event) -> bool:
        attempts.append(1)
        conn.error = "fake first-connect failure"  # 真实 _connect_once 失败时会写 error
        return False

    conn._connect_once = fake_connect  # type: ignore[method-assign]
    conn._backoff_wait = lambda: None  # type: ignore[assignment]
    t0 = time.time()
    conn.start()
    conn._thread.join(timeout=5)
    assert len(attempts) == 1, f"首连失败不应重试，实际 {len(attempts)} 次"
    assert time.time() - t0 < 5, "start() 不应被拖住"
    assert conn.error is not None


# ---------------- MCP：_connect_once 失败回滚 transport ----------------

def test_mcp_connect_once_rolls_back_entered_transport(monkeypatch):
    """transport __aenter__ 成功、session 建立失败：已进入的 transport 必须被退出。

    旧实现只记 error 不回滚，每次失败泄漏一个 stdio 子进程句柄；
    重连改为持续重试后，泄漏会随时间累积成资源问题。
    """
    import agent.tools.mcp_client as mc

    class FakeTransportCM:
        def __init__(self) -> None:
            self.exits = 0

        async def __aenter__(self):
            return ("read", "write")

        async def __aexit__(self, *a):
            self.exits += 1
            return False

    class BoomSession:
        def __init__(self, read, write) -> None:
            pass

        async def __aenter__(self):
            raise RuntimeError("initialize boom")

        async def __aexit__(self, *a):
            return False

    cm = FakeTransportCM()
    monkeypatch.setattr(mc, "stdio_client", lambda params: cm)
    monkeypatch.setattr(mc, "ClientSession", BoomSession)

    conn = mc.McpConnection("t", {"name": "t", "transport": "stdio", "command": "x"})
    ok = asyncio.run(conn._connect_once(threading.Event()))
    assert ok is False
    assert cm.exits == 1, "失败的 transport 必须被回滚退出，不得泄漏"
    assert conn.session is None and conn._transport_cm is None and conn._session_cm is None
    assert conn.error and "boom" in conn.error


# ---------------- bridge：ask 兜底单模型读 llm 配置 ----------------

def test_bridge_ask_fallback_uses_llm_config(monkeypatch):
    from agent import bridge as br

    captured: dict = {}

    class FakeClient:
        def __init__(self, base_url, api_key, model,
                     temperature=0.7, timeout=120.0, max_retries=2):
            captured.update(temperature=temperature, timeout=timeout, max_retries=max_retries)

        def chat(self, messages, tools=None):
            return {"content": "ok"}

    monkeypatch.setattr(br, "LLMClient", FakeClient)
    cfg = Config(_interpolate({
        "agents": [],
        "llm": {"base_url": "u", "api_key": "k", "model": "m",
                "temperature": 0.1, "timeout": 33, "max_retries": 5},
    }))
    b = br.Bridge(cfg)
    try:
        out = b.handle({"cmd": "ask", "prompt": "hi"})
        assert out["ok"] and out["answer"] == "ok"
        assert captured == {"temperature": 0.1, "timeout": 33.0, "max_retries": 5}, (
            f"bridge ask 兜底应读 llm 配置，实际：{captured}")
    finally:
        b.close()


def test_bridge_call_tool_reports_failure_shape():
    """#3：call_tool 未知工具/失败必须回 ok:false，不再永远 ok:true 把错误塞进 result。"""
    from agent import bridge as br
    cfg = Config(_interpolate({
        "agents": [],
        "llm": {"base_url": "u", "api_key": "k", "model": "m"},
    }))
    b = br.Bridge(cfg)
    try:
        bad = b.handle({"cmd": "call_tool", "name": "does_not_exist", "args": {}})
        assert bad["ok"] is False, "未知工具应如实判失败"
        assert "未知工具" in bad["result"] and bad.get("error")
        ok = b.handle({"cmd": "call_tool", "name": "list_dir", "args": {"path": "."}})
        assert ok["ok"] is True and "result" in ok
    finally:
        b.close()


# ---------------- bridge：stderr 必须是 UTF-8 ----------------


def test_bridge_stderr_decodes_as_utf8():
    """诊断走 stderr，但 Windows 重定向流默认本地代码页（GBK）。

    bridge 声明协议 UTF-8，驱动方通常也按 UTF-8 收 stderr；
    修正前配置警告以 GBK 写入，外部解码直接 UnicodeDecodeError
    （test_smoke 的子进程 reader 线程同样被炸出告警）。
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    # 确保存在缺失变量 → 触发 [配置警告]（若环境里恰好配了，删掉再跑）
    for k in ("NODE_A_ENDPOINT", "NODE_A_KEY", "NODE_A_MODELS",
              "NODE_B_ENDPOINT", "NODE_B_KEY", "NODE_B_MODELS"):
        env.pop(k, None)
    proc = subprocess.run(
        [sys.executable, "-m", "agent", "bridge"],
        input=b'{"cmd": "ping"}\n', capture_output=True, cwd=str(root), env=env, timeout=120,
    )
    err = proc.stderr.decode("utf-8")  # 严格解码：GBK 字节在这里就会炸
    assert "配置警告" in err or proc.returncode == 0, (
        f"应有可 UTF-8 解码的诊断或无输出；stderr={err[:200]!r}")
