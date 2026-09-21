"""MAO 冒烟测试：全部离线，无需 API key / 网络。

覆盖：
- 配置解析（load_config）
- 内置工具注册（build_builtin_tools）
- WorkerPool 并发派工不崩（验证数据竞争修复，任务1）
- Web 会话解析：session_id 校验、选主换主即重建会话（本轮修正）
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.config import Config, _interpolate, load_config
from agent.core.orchestrator import WorkerPool
from agent.tools.builtin import build_builtin_tools


def test_config_loads():
    cfg = load_config()
    assert cfg is not None
    assert isinstance(cfg.agent_profiles, list)
    # collaboration 默认值应可读、合法
    assert cfg.max_workers >= 1
    assert cfg.max_iterations >= 1


def test_builtin_tools_registered():
    cfg = load_config()
    reg = build_builtin_tools(cfg)
    schemas = reg.openai_schemas()
    assert len(schemas) >= 1
    names = {s["function"]["name"] for s in schemas}
    # 至少应注册内置 shell 工具
    assert "run_shell" in names


def test_worker_pool_construction():
    cfg = load_config()
    pool = WorkerPool(cfg, exclude=None)
    # names() 返回当前池内的工人名列表（可能为空的 solo）
    assert isinstance(pool.names(), list)


def test_worker_pool_concurrent_ask_no_crash():
    """并发派工不应因共享状态（LRU 缓存 / 节点健康）竞争而抛异常（任务1）。"""
    cfg = load_config()
    pool = WorkerPool(cfg, exclude=None)
    if not pool.names():
        # 池为空（solo 配置）时仅验证构造与 names() 不崩即可
        assert pool.names() == []
        return

    # 用假 LLMClient 替代真实网络调用
    import agent.core.orchestrator as orch

    fake = MagicMock()
    fake.chat.return_value = {"role": "assistant", "content": "ok"}

    def make_fake_client(*_args, **_kwargs):
        return fake

    orig = orch.LLMClient
    orch.LLMClient = make_fake_client
    try:
        worker = pool.names()[0]
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(pool.ask, worker, f"task-{i}") for i in range(20)]
            for f in futs:
                assert f.result(timeout=10) == "ok"
        # 并发写入后，缓存命中仍返回一致结果（验证锁保护了 LRU）
        assert pool.ask(worker, "task-0") == "ok"
    finally:
        orch.LLMClient = orig


def test_web_chat_session_resolution(monkeypatch):
    """Web /api/chat 的会话解析逻辑（本轮修正）：

    1) 非法 session_id 直接报错且不构造 Agent（原本可拼出路径穿越的落盘文件名）；
    2) 同一 session_id 换主智能体 → 旧 Agent 关闭、按新主重建
       （原本复用旧 Agent，新的 orchestrator 被静默忽略，顶栏切换主不生效）。

    用 FakeAgent 替掉真实 Agent，全程离线；SSE 生成器不消费，只断言会话表状态。
    """
    import web.server as ws

    cfg = Config(_interpolate({"agents": [
        {"name": "n1", "base_url": "u", "api_key": "k", "model": "m1"},
        {"name": "n2", "base_url": "u", "api_key": "k", "model": "m2"},
    ]}))
    monkeypatch.setattr(ws, "_get_cfg", lambda: cfg)

    built: list = []

    class FakeAgent:
        mcp_status: list = []

        def __init__(self, cfg, session_id=None, profile=None, **kw):
            self.session_id = session_id
            self.profile = profile
            self.registry = SimpleNamespace(names=lambda: [])
            self.closed = False
            built.append(self)

        def run(self, message, on_event=None, should_stop=None):
            return "ok"

        def close(self):
            self.closed = True

    monkeypatch.setattr(ws, "Agent", FakeAgent)
    monkeypatch.setattr(ws, "_agents", {})
    monkeypatch.setattr(ws, "_last_active", {})
    monkeypatch.setattr(ws, "_session_orch", {})

    # 1) 非法 session_id：直接报错，不构造 Agent
    resp = ws.chat(ws.ChatRequest(message="hi", session_id="../../x"))
    assert isinstance(resp, dict) and "非法" in resp["error"]
    assert built == []

    # 2) 指定主智能体 → 会话绑定该主
    ws.chat(ws.ChatRequest(message="hi", session_id="s1", orchestrator="n2"))
    assert ws._session_orch["s1"] == "n2"
    first = ws._agents["s1"]
    assert first.profile.name == "n2"

    # 3) 同会话同主 → 复用同一 Agent
    ws.chat(ws.ChatRequest(message="hi", session_id="s1", orchestrator="n2"))
    assert ws._agents["s1"] is first
    assert first.closed is False

    # 4) 同会话换主 → 旧 Agent 关闭、按新主重建（切换不再静默失效）
    ws.chat(ws.ChatRequest(message="hi", session_id="s1", orchestrator="n1"))
    assert first.closed is True, "换主后旧会话 Agent 应被关闭"
    assert ws._agents["s1"] is not first
    assert ws._agents["s1"].profile.name == "n1"


def test_bridge_stdout_is_pure_json_lines():
    """真实子进程验证 bridge 协议：stdout 每一行都必须是可解析的 JSON。

    实测修正前，配置缺失变量的 `[配置警告]` 会 print 到 stdout，与 ready/响应行
    混在一起，按行 json.loads 的外部智能体驱动方在启动阶段就会解析失败。
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "-m", "agent", "bridge"],
        input=json.dumps({"cmd": "ping"}, ensure_ascii=False) + "\n",
        capture_output=True, text=True, encoding="utf-8", cwd=str(root), timeout=120,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines, f"bridge 无输出；stderr={proc.stderr[:400]}"

    parsed = []
    for ln in lines:
        try:
            parsed.append(json.loads(ln))
        except json.JSONDecodeError as e:  # noqa: PERF203
            raise AssertionError(f"stdout 混入非 JSON 行：{ln!r}") from e

    # 首行是 ready 事件，末行是 ping 的响应
    assert parsed[0].get("event") == "ready"
    assert parsed[-1].get("ok") is True
