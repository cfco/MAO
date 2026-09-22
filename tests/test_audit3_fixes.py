"""审计 P0/P1/P2 批次（2026-09-22 第二轮全量排查）的回归测试：全部离线。

对应问题（编号见对话审计报告）：
- P0  McpTool.execute 少一个 ctx 形参 —— 注册表恒以 execute(args, ctx) 调用，
  所有 mcp__* 工具 100% TypeError，被兜底 except 吞成"工具执行出错"。
- P1a ask_many/run_review 静默丢弃未知工人 —— 文档承诺的 missing 状态码从不出现。
- P1b MCP 工具重名注册抛 ValueError —— 在 Bridge.__init__ 里会炸掉整个进程。
- P1c vote 的 threshold 不校验类型 —— 字符串阈值直通 `ratio >= th` 抛 TypeError。
- P2a MCP 会话 CM 跨 task 退出 —— anyio cancel scope 必抛错被吞，重连每轮泄漏句柄。
- P2b load_config 显式 path 与全局缓存互污。
- P2c 批量派工线程非 daemon —— 解释器退出被在飞 LLM 请求拖住。
- P2d read_file 的 limit 裸 int() —— 模型传坏值直接炸进兜底异常路径。
"""
from __future__ import annotations

import asyncio
import threading

import agent.config as config_mod
import agent.tools.mcp_client as mc
from agent.config import Config, _interpolate
from agent.core import orchestrator as orch
from agent.core.orchestrator import WorkerPool
from agent.tools.base import ToolRegistry, ToolResult


def _pool(models: str, cap: int = 3) -> WorkerPool:
    cfg = Config(_interpolate({
        "agents": [{"name": "st", "base_url": "u", "api_key": "k", "models": models}],
        "collaboration": {"max_participants": cap, "max_workers": 2},
    }))
    return WorkerPool(cfg, exclude=None)


class _InstantClient:
    def __init__(self, *_a, **_k):
        pass

    def chat(self, messages, tools=None):
        return {"role": "assistant", "content": "hi"}

    def close(self):
        pass


# ---------------- P0：MCP 工具必须能经注册表执行 ----------------

def test_mcp_tool_executes_through_registry():
    """P0 复现转正：注册表调用 McpTool 不再 TypeError，成功/失败契约如实透传。"""

    class FakeConn:
        def __init__(self):
            self.calls = []

        def call_tool(self, remote_name, args):
            self.calls.append((remote_name, args))
            return f"echo:{remote_name}:{args.get('msg', '')}"

    conn = FakeConn()
    tool = mc.McpTool(name="mcp__fake__echo", description="d",
                      input_schema={}, conn=conn, remote_name="echo")
    reg = ToolRegistry()
    reg.register(tool)

    res = reg.run("mcp__fake__echo", {"msg": "hi"})
    assert res["ok"] is True, f"MCP 工具经注册表必须成功，实际：{res}"
    assert res["result"] == "echo:echo:hi"
    assert conn.calls == [("echo", {"msg": "hi"})]

    # 失败路径：ToolResult(ok=False) 必须映射成 ok:false + error
    conn.call_tool = lambda *_a: ToolResult("错误：远端失败", ok=False)  # type: ignore[method-assign]
    bad = reg.run("mcp__fake__echo", {})
    assert bad["ok"] is False and "远端失败" in bad["error"]


# ---------------- P1a：未知工人显式 missing ----------------

def test_ask_many_structured_surfaces_missing_worker(monkeypatch):
    monkeypatch.setattr(orch, "LLMClient", _InstantClient)
    # 用两模型站：单模型站按命名规则就叫 "st"（不带 :模型 后缀），
    # 这里要的是"点名存在的工人 + 点名不存在的工人"两种状态并存
    pool = _pool("m1,m2")
    recs = pool.ask_many_structured(["st:m1", "ghost"], "任务")
    by = {r["worker"]: r["status"] for r in recs}
    assert by == {"st:m1": "ok", "ghost": "missing"}, (
        "点名的未知工人必须逐条带回 missing（旧实现静默丢弃 → 外部主拿错名字无限重试）")
    ghost = next(r for r in recs if r["worker"] == "ghost")
    assert ghost["ok"] is False and ghost["answer"] == ""


def test_bridge_ask_many_all_ghost_fails_with_hint():
    """全部点名都是错名字：外层 ok:false，但逐工人结果必须能看出是 missing。"""
    from agent import bridge as br

    orig = orch.LLMClient
    orch.LLMClient = _InstantClient  # 离线：不真发请求
    try:
        cfg = Config(_interpolate({"agents": [
            {"name": "st", "base_url": "u", "api_key": "k", "model": "m"},
        ]}))
        b = br.Bridge(cfg)
        try:
            out = b.handle({"cmd": "ask_many", "prompt": "p", "workers": ["typo-name"]})
            assert out["ok"] is False
            assert [w["status"] for w in out["workers"]] == ["missing"]
        finally:
            b.close()
    finally:
        orch.LLMClient = orig


# ---------------- P1b：MCP 重名跳过、不炸 Bridge ----------------

def test_duplicate_mcp_tool_names_skipped_not_fatal(monkeypatch):
    class T:
        def __init__(self, name):
            self.name = name
            self.inputSchema = {}
            self.description = ""

    class FakeConn:
        def __init__(self, name, cfg):
            self.name = name
            self.error = None
            self.session = object()  # 假装已连接

        def start(self):
            pass

        def list_tools(self):
            return [T("x")]

        def stop(self):
            pass

    monkeypatch.setattr(mc, "McpConnection", FakeConn)
    reg = ToolRegistry()
    # "a.b" 与 "a_b" 经 _safe_name 归一后同为 "a_b" → 第二个工具必须被跳过
    # 而不是 ValueError 炸穿（_safe_name 保留连字符，故不能用 "a-b" 构造冲突）
    status = mc.connect_and_register([{"name": "a.b"}, {"name": "a_b"}], reg)
    assert reg.names() == ["mcp__a_b__x"], f"重名应只留第一个：{reg.names()}"
    assert any("跳过重名" in s for s in status), f"状态行要如实报告跳过：{status}"


# ---------------- P1c：threshold 类型归一 ----------------

def test_vote_threshold_string_coerced(monkeypatch):
    pool = _pool("m1,m2")

    def spy(worker, prompt, system=None):
        # 投票阶段（ballot 提示词）统一投 1 号；收集阶段给方案文本
        ans = "1" if "请只输出你认可方案的编号" in prompt else "方案内容"
        return {"worker": worker, "ok": True, "status": "ok", "answer": ans, "error": ""}

    monkeypatch.setattr(pool, "ask_result", spy)
    v = pool.vote("选一个", workers=pool.names(), threshold="0.5")  # 字符串阈值
    assert v["ok"] is True and v["consensus"] is True, (
        '"0.5" 应归一为 0.5 判过半共识（旧实现 ratio >= "0.5" 直接 TypeError）')
    v2 = pool.vote("选一个", workers=pool.names(), threshold="half")  # 非法值回退默认
    assert v2["ok"] is True and v2["consensus"] is True


# ---------------- P2a：会话 CM 同 task 进入 / 退出 ----------------

def test_mcp_session_cm_entered_and_exited_in_same_task(monkeypatch):
    seen: dict[str, object] = {}

    class TransportCM:
        async def __aenter__(self):
            seen["enter"] = asyncio.current_task()
            return ("read", "write")

        async def __aexit__(self, *a):
            seen["exit"] = asyncio.current_task()
            return False

    class SessionCM:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def initialize(self):
            pass

    monkeypatch.setattr(mc, "stdio_client", lambda params: TransportCM())
    monkeypatch.setattr(mc, "ClientSession", lambda r, w: SessionCM())

    conn = mc.McpConnection("t", {"name": "t", "transport": "stdio", "command": "x"})
    conn.start()
    assert conn.session is not None or conn.error is None  # 连上了（或至少没报错）
    conn.stop()
    conn._thread.join(timeout=10)
    assert not conn._thread.is_alive(), "stop 后线程应干净退出"
    assert "enter" in seen and "exit" in seen, (
        "会话 CM 必须被真正退出（旧跨 task 版本 __aexit__ 必抛错被吞，等于从不关闭）")
    assert seen["enter"] is seen["exit"], "enter/exit 必须同 task（anyio cancel scope 约束）"


# ---------------- P2b：显式 path 不进全局缓存 ----------------

def test_explicit_path_load_config_bypasses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "ROOT", tmp_path)
    a = tmp_path / "a.yaml"
    a.write_text('agents: [{name: sa, base_url: u, api_key: k, model: m1}]\n', encoding="utf-8")
    b = tmp_path / "b.yaml"
    b.write_text('agents: [{name: sb, base_url: u, api_key: k, model: m2}]\n', encoding="utf-8")
    cfg_a = config_mod.load_config(a)
    cfg_b = config_mod.load_config(b)
    assert [p.name for p in cfg_a.agent_profiles] == ["sa"]
    assert [p.name for p in cfg_b.agent_profiles] == ["sb"], (
        "显式 path 每次真实读盘：旧实现第二次拿到的是 a.yaml 的缓存")
    # 默认路径的缓存行为不变
    (tmp_path / "config.yaml").write_text(
        'agents: [{name: sd, base_url: u, api_key: k, model: m4}]\n', encoding="utf-8")
    c1 = config_mod.load_config()
    assert config_mod.load_config() is c1, "默认路径无变化仍应命中缓存"


# ---------------- P2c：批量派工线程 daemon 化 ----------------

def test_run_parallel_uses_daemon_threads(monkeypatch):
    pool = _pool("m1,m2")
    flags: list[bool] = []

    def spy(worker, prompt, system=None):
        flags.append(threading.current_thread().daemon)
        return {"worker": worker, "ok": True, "status": "ok", "answer": "a", "error": ""}

    monkeypatch.setattr(pool, "ask_result", spy)
    pool.ask_many(["st:m1", "st:m2"], "q")
    assert flags and all(flags), (
        "派工线程必须是 daemon：ThreadPoolExecutor 的非 daemon 线程会被解释器 "
        "atexit join，进程收尾被在飞 LLM 请求拖住")


# ---------------- P2d：read_file 的 limit 容错 ----------------

def test_read_file_bad_limit_falls_back_not_crashes(tmp_path, monkeypatch):
    from agent.tools import builtin

    monkeypatch.setattr(builtin, "ROOT", tmp_path)
    monkeypatch.setattr(builtin, "ROOT_RESOLVED", tmp_path.resolve())
    monkeypatch.setattr(builtin, "_WORKSPACE_ROOTS", (tmp_path.resolve(),))
    reg = builtin.build_builtin_tools(Config({}))
    (tmp_path / "f.txt").write_text("第一行\n第二行\n第三行\n", encoding="utf-8")

    out = reg.run("read_file", {"path": "f.txt", "limit": "many"})
    assert out["ok"] is True and "第一行" in out["result"], (
        f"limit 类型错应回退默认值继续读，实际：{out}")
    out0 = reg.run("read_file", {"path": "f.txt", "limit": -5})
    assert out0["ok"] is True and "第一行" in out0["result"], "负数 limit 钳到 ≥1，不得砍尾"
