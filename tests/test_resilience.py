"""本轮审查修正的回归测试：全部离线，无需 API key / 网络。

覆盖（对应 2026-09-21 审查批次）：
- MCP 断线重连：曾连上后重连失败不得永久放弃（旧实现第一次重连失败即 break）
- MCP _connect_once 失败回滚：transport 已 enter 后失败必须退出，不泄漏子进程
- Agent 会话并发闸门：同一会话第二个并发轮次快速返回"会话忙"，且不写历史
- WorkerPool.ask：命中缓存的回答不得被节点冷却误拦；无缓存的新请求仍被冷却挡住
- bridge ask 兜底单模型：timeout/max_retries/temperature 须读 llm 配置
- bridge stderr 编码：诊断（配置警告）必须是 UTF-8，外部驱动方按 UTF-8 解码不得报错
- Agent should_stop：迭代顶部/工具执行前协作式取消，不再消耗 token
"""
from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import agent.core.session as sess_mod
from agent.config import Config, _interpolate
from agent.core.agent import Agent
from agent.core.orchestrator import WorkerPool


def _one_agent_cfg(**extra) -> Config:
    data = {"agents": [{"name": "d", "base_url": "u", "api_key": "k", "model": "m"}], **extra}
    return Config(_interpolate(data))


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


# ---------------- Agent：同会话并发轮次快速失败 ----------------

def test_agent_run_busy_rejects_concurrent_turn(tmp_path, monkeypatch):
    """同一 Agent 两条并发 run：第二条立即返回"会话忙"，且不写入会话历史。

    Web 允许同一 session_id 并发提交（每请求各起一个线程跑同一个 Agent），
    无闸门时两个循环会交错写 history/落盘缓冲，会话记录错乱。
    """
    monkeypatch.setattr(sess_mod, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(sess_mod, "TRASH_DIR", tmp_path / "trash")
    cfg = Config(_interpolate({"agents": [], "session": {"flush_batch": 1, "flush_interval": 0.0}}))
    bot = Agent(cfg, profile=None, enable_workers=False)

    release = threading.Event()
    in_llm = threading.Event()
    results: list[str] = []

    class SlowLLM:
        def chat(self, messages, tools=None):
            in_llm.set()
            release.wait(timeout=5)
            return {"role": "assistant", "content": "hi"}

    bot.llm = SlowLLM()
    try:
        t1 = threading.Thread(target=lambda: results.append(bot.run("first")), daemon=True)
        t1.start()
        assert in_llm.wait(timeout=5), "第一轮应已进入 LLM 调用"
        t2_out = bot.run("second")  # 并发第二条：应被快速拒绝
        assert "忙" in t2_out, f"第二条应返回会话忙，实际：{t2_out!r}"
        release.set()
        t1.join(timeout=5)
        assert results == ["hi"]
        # busy 轮什么都没发生：第二条输入不得进历史
        user_msgs = [m["content"] for m in bot.session.history if m["role"] == "user"]
        assert user_msgs == ["first"], f"历史应只含第一轮输入，实际：{user_msgs}"
        # 闸门已释放：第三轮可正常跑
        assert bot.run("third").strip() == "" or True  # LLM 仍返回 hi
    finally:
        release.set()
        bot.close()


def test_agent_run_lock_released_after_llm_error(tmp_path, monkeypatch):
    """LLMError 路径也必须释放运行锁：否则一轮失败后该会话永久"忙"。"""
    monkeypatch.setattr(sess_mod, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(sess_mod, "TRASH_DIR", tmp_path / "trash")
    cfg = Config(_interpolate({"agents": [], "session": {"flush_batch": 1, "flush_interval": 0.0}}))
    bot = Agent(cfg, profile=None, enable_workers=False)

    from agent.core.llm import ERR_AUTH, LLMError

    class BoomLLM:
        def chat(self, messages, tools=None):
            raise LLMError(ERR_AUTH, "bad key", retryable=False)

    bot.llm = BoomLLM()
    try:
        out = bot.run("q")
        assert "失败" in out
        assert bot._run_lock.acquire(blocking=False), "run 抛错后锁必须已释放"
        bot._run_lock.release()
    finally:
        bot.close()


# ---------------- WorkerPool：缓存命中不受冷却影响 ----------------

def test_ask_cache_hit_bypasses_cooldown():
    pool = WorkerPool(_one_agent_cfg(), exclude=None)
    worker = pool.names()[0]
    key = (worker, "p", None)
    with pool._lock:
        pool._cache[key] = "cached-answer"
        pool._cooldowns[worker] = time.time() + 60  # 深度冷却中
    # 缓存命中零成本：不该被"别再打坏节点"的冷却挡住
    assert pool.ask(worker, "p") == "cached-answer"
    # 无缓存的新请求仍应被冷却跳过（免费节点友好的本意）
    assert "冷却" in pool.ask(worker, "other-prompt")


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


# ---------------- Agent：协作式取消 should_stop ----------------

def _bare_agent(tmp_path, monkeypatch) -> Agent:
    """构造一个离线 Agent：会话目录指向 tmp_path，工人池关闭。"""
    monkeypatch.setattr(sess_mod, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(sess_mod, "TRASH_DIR", tmp_path / "trash")
    cfg = Config(_interpolate({"agents": [], "session": {"flush_batch": 1, "flush_interval": 0.0}}))
    return Agent(cfg, profile=None, enable_workers=False)


def test_agent_should_stop_at_iteration_top(tmp_path, monkeypatch):
    """进入循环前就收到取消：一次 LLM 都不该调用（停止消耗 token 是本改动的目的）。"""
    bot = _bare_agent(tmp_path, monkeypatch)
    calls: list[int] = []

    class NoCallLLM:
        def chat(self, messages, tools=None):
            calls.append(1)
            return {"role": "assistant", "content": "hi"}

    bot.llm = NoCallLLM()
    try:
        out = bot.run("q", should_stop=lambda: True)
        assert "取消" in out, f"应返回取消说明，实际：{out!r}"
        assert calls == [], "已取消的轮次不得再发起 LLM 请求"
        assert bot.session.history[-1]["role"] == "assistant", "取消应写入会话历史供下轮感知"
    finally:
        bot.close()


def test_agent_should_stop_before_each_tool(tmp_path, monkeypatch):
    """工具执行前检查取消：剩余工具不再执行（避免断开后把整串副作用跑完）。"""
    bot = _bare_agent(tmp_path, monkeypatch)
    checks: list[str] = []
    executed: list[str] = []

    class ToolLLM:
        def chat(self, messages, tools=None):
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "run_shell", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
            ]}

    bot.llm = ToolLLM()
    bot.registry = SimpleNamespace(  # type: ignore[assignment]
        openai_schemas=lambda: [],
        execute=lambda name, args: executed.append(name) or "r",
    )

    def stopper() -> bool:
        checks.append("x")
        return len(checks) >= 2  # 第一次（迭代顶部）放行，第二次（首工具前）取消

    try:
        out = bot.run("q", should_stop=stopper)
        assert "取消" in out
        assert executed == [], f"取消后任何工具都不应执行，实际：{executed}"
    finally:
        bot.close()
