"""第六轮审计（2026-10-19）回归：只收本轮行为变化，按模块归类、不按轮次堆。

覆盖三件事：
1. 兜底未知异常（ERR_OTHER）**不进当日隔离**，只算一次失败（浓度到阈值仍进冷却）；
   而 auth / 非重试 4xx 照旧隔离——防止把"重试没用"与"模型坏了"混为一谈。
2. `_child_env` 连**紧凑命名**的密钥变量（MYKEY / GHTOKEN）一起剔除：旧实现只认
   `_KEY` 这类分段后缀，`MYKEY=` 会原样进子进程环境，一条 `set` 就抄走。
3. 覆盖留档目录在测试期已隔离（conftest.isolate_backup_dir），守卫见 test_repo_hygiene。
"""

from __future__ import annotations

import os

from agent.config import Config, _interpolate
from agent.core import orchestrator as orch
from agent.core.llm import ERR_OTHER, LLMError
from agent.core.orchestrator import WorkerPool
from agent.tools import builtin


def _pool() -> WorkerPool:
    """两站小池（st: m1/m2，other: x1），够跑失败分类，不拉起任何真实档案。"""
    cfg = Config(
        _interpolate(
            {
                "agents": [
                    {"name": "st", "base_url": "u1", "api_key": "k", "models": ["m1", "m2"]},
                    {"name": "other", "base_url": "u2", "api_key": "k2", "model": "x1"},
                ],
                "collaboration": {"fallback_cross_station": False},
            }
        )
    )
    return WorkerPool(cfg, exclude=None)


def test_unknown_exception_only_cools_not_quarantines(monkeypatch):
    """ERR_OTHER（SDK 兜底/解析失败等）不该把一个健康节点封一整天。"""

    class UnknownBoom:
        def __init__(self, base_url, api_key, model, **kw):
            self.model = model

        def chat_once(self, messages):
            raise LLMError(ERR_OTHER, "unexpected sdk failure", retryable=False)

    monkeypatch.setattr(orch, "LLMClient", UnknownBoom)
    pool = _pool()
    pool._max_retries = 0  # 本用例只验记账口径，不叠加重试

    r1 = pool.ask_result("st:m1", "任务")
    assert r1["status"] == "error"
    assert not pool.health.quarantined("st:m1"), (
        "兜底未知异常被当成终态失败 → 整个模型当天隔离；应只按普通失败记账"
    )

    # 第二次仍应真实派工（返回 error），而不是被跳过成 quarantined
    r2 = pool.ask_result("st:m1", "任务")
    assert r2["status"] == "error", f"未知异常不得导致当日隔离，实际 status={r2['status']}"


def test_auth_error_still_quarantines(monkeypatch):
    """回归保护：放宽 ERR_OTHER 不能顺手把 auth 这类真正的终态错误也放过。"""

    class AuthBoom:
        def __init__(self, base_url, api_key, model, **kw):
            self.model = model

        def chat_once(self, messages):
            raise LLMError("auth", "bad key", retryable=False)

    monkeypatch.setattr(orch, "LLMClient", AuthBoom)
    pool = _pool()
    pool._max_retries = 0

    r = pool.ask_result("st:m1", "任务")
    assert r["status"] == "error"
    assert pool.health.quarantined("st:m1"), "认证失败仍必须当日隔离（整站级问题）"


def test_child_env_strips_compact_secret_names(monkeypatch):
    """紧凑命名（无下划线分隔）的密钥变量同样不得进子进程环境。"""

    monkeypatch.setenv("MYKEY", "v1")
    monkeypatch.setenv("GHTOKEN", "v2")
    monkeypatch.setenv("OPENAIKEY", "v3")
    monkeypatch.setenv("NODE_A_KEY", "v4")
    monkeypatch.setenv("MONKEY", "v5")
    env = builtin._child_env()

    for name in ("MYKEY", "GHTOKEN", "OPENAIKEY", "NODE_A_KEY"):
        assert name not in env, f"{name} 是密钥形态，必须从子进程环境中剔除"
    # MONKEY 属同尾词误剔：安全侧**刻意宁严勿漏**（子进程少一个无关变量，
    # 远轻于把真密钥交给工具输出）。这条断言把该取舍锁住，别悄悄改成"精确匹配"。
    assert "MONKEY" not in env
    # 功能变量必须保留，否则 cmd 跑不起来
    assert "PATH" in env or os.name != "nt"


def test_gate_dir_is_redirected_in_tests():
    """测试期的留档目录必须在仓库外（否则每跑一次就多几个 .bak）。"""
    from agent.config import ROOT

    assert ROOT not in builtin.BACKUP_DIR.parents, (
        f"BACKUP_DIR 指向仓库内：{builtin.BACKUP_DIR}，conftest 的隔离没生效"
    )


def test_removed_worker_clients_are_reclaimed(monkeypatch):
    """被移出池的工人：客户端摘出缓存并**延迟关闭**（不掐在飞请求），空站的门被摘掉。"""
    closed: list[str] = []

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def chat_once(self, messages):
            return {"role": "assistant", "content": "x"}

        def close(self):
            closed.append("close")

        def close_when_idle(self):
            closed.append("idle")

    monkeypatch.setattr(orch, "LLMClient", FakeClient)
    pool = _pool()
    m1, m2 = pool.profiles["st:m1"], pool.profiles["st:m2"]
    pool._client_for(m1)
    pool._client_for(m2)
    pool._station_gate(m1.station_key)
    assert set(pool._clients) == {"st:m1", "st:m2"}

    # m2 还在 → 同站的门必须保留（否则等于给同站偷偷重置了并发配额）
    pool._retire_worker_clients(["st:m1"], {m2.station_key})
    assert "st:m1" not in pool._clients, "被移出的工人不应留在客户端缓存里"
    assert "st:m2" in pool._clients, "仍在池里的工人不能被误回收"
    assert m2.station_key in pool._station_gates
    assert closed == ["idle"], f"应走延迟关闭而不是直接 close，实际 {closed}"

    # 站里已无任何模型 → 门对象摘掉（等待中的线程仍持有旧引用，会正常跑完）
    pool._retire_worker_clients(["st:m2"], set())
    assert pool._station_gates == {}, f"空站的站门应被摘掉，实际 {pool._station_gates}"


def test_sync_free_nodes_reclaims_removed_worker(monkeypatch):
    """热同步移除工人时真的触发回收，而不是只改 profiles 就完事。"""
    calls: list[list[str]] = []
    pool = _pool()
    monkeypatch.setattr(
        pool, "_retire_worker_clients", lambda removed, live: calls.append(sorted(removed))
    )
    # _pool() 的 models 是字面量 → models_env 为空串，用空串键即可命中 retired 分支
    out = pool.sync_free_nodes({"": {"retired": ["m1"], "added": []}})
    assert out["removed"] == ["st:m1"]
    assert "st:m1" not in pool.profiles
    assert calls == [["st:m1"]], "移除工人后没有回收其连接池"


# ---------------- 语义四项（2026-10-19）：可配化 / 自适应降级 ----------------


def test_vote_preview_length_is_configurable(monkeypatch):
    """投票预览长度读 collaboration.vote_preview_chars，不再是写死的 80/96 字。"""
    cfg = Config(
        _interpolate(
            {
                "agents": [{"name": "st", "base_url": "u1", "api_key": "k", "models": ["m1"]}],
                "collaboration": {"vote_preview_chars": 300},
            }
        )
    )
    pool = WorkerPool(cfg, exclude=None)
    captured: dict = {}

    def fake_run(workers, prompt, system, timeout, on_done=None):
        captured["prompt"] = prompt
        return {}  # 无票：_run_ballot 返回 None，但我们只关心它发出去的候选预览

    monkeypatch.setattr(pool, "_run_parallel", fake_run)
    pool._run_ballot([("st:m1", "x" * 500)], ["st:m1"], 10)

    ballot = captured["prompt"]
    assert "x" * 300 in ballot, "预览长度没跟着配置走（仍是旧的 80/96 字截断）"
    assert "x" * 301 not in ballot, "预览超出配置长度"

    # 边界：配置过小/过大时被夹到 40~2000
    small = Config(_interpolate({"collaboration": {"vote_preview_chars": 1}}))
    big = Config(_interpolate({"collaboration": {"vote_preview_chars": 99999}}))
    assert small.vote_preview_chars == 40
    assert big.vote_preview_chars == 2000


def test_mcp_tool_timeout_is_configurable():
    """MCP 工具调用超时读 tools.mcp_tool_timeout，不再硬编码 300s；超时进连接组签名。"""
    from agent.tools import mcp_client

    cfg = Config(_interpolate({"tools": {"mcp_tool_timeout": 42}}))
    assert cfg.mcp_tool_timeout == 42

    conn = mcp_client.McpConnection("t", {"name": "t"}, cfg.mcp_tool_timeout)
    assert conn._call_timeout == 42

    servers = [{"name": "t", "transport": "stdio"}]
    assert mcp_client._group_key(servers, 42) != mcp_client._group_key(servers, 300), (
        "超时不同的两个使用者不能共用连接组（先到者的超时会影响后到者）"
    )


def test_probe_degrades_to_no_max_tokens_per_station(monkeypatch):
    """探测带 max_tokens 遇到"参数不支持"类 4xx：按站降级重测一次，而不是整轮丢样本。"""

    class PickyClient:
        def __init__(self):
            self.calls: list[dict] = []

        def chat_once(self, messages, max_tokens=None):
            self.calls.append({"max_tokens": max_tokens})
            if max_tokens is not None:
                raise LLMError("api", "HTTP 400: unknown parameter max_tokens", retryable=False)
            return {"role": "assistant", "content": "pong"}

    pool = _pool()
    pool._latency_probe_max_tokens = 8
    client = PickyClient()
    monkeypatch.setattr(pool, "_probe_client_for", lambda p: client)

    ms = pool._probe_one("st:m1")
    station = pool.profiles["st:m1"].station_key
    assert ms is not None, "降级重测失败，说明 4xx 把整次探测打死了"
    assert station in pool._probe_no_max_tokens
    assert [c["max_tokens"] for c in client.calls] == [8, None], (
        f"应先带参数失败、再不带参数重测，实际调用序列 {client.calls}"
    )

    # 同一站第二次探测直接用不带参数的口径，不再白打一次失败请求
    client.calls.clear()
    assert pool._probe_one("st:m1") is not None
    assert [c["max_tokens"] for c in client.calls] == [None]


def test_named_workers_over_cap_warns_but_does_not_block(capsys):
    """显式点名超过 max_participants：只提示不阻断（决定权留给主）。"""
    from agent.bridge import _warn_oversized_named_list

    cfg = Config(_interpolate({"collaboration": {"max_participants": 2}}))
    _warn_oversized_named_list(cfg, ["a", "b", "c"])
    err = capsys.readouterr().err
    assert "超过 collaboration.max_participants=2" in err
    assert "3 个工人" in err

    _warn_oversized_named_list(cfg, ["a", "b"])  # 未超限：不吵
    assert capsys.readouterr().err == ""

    _warn_oversized_named_list(cfg, None)  # 缺省名单（走 pick）：也不该告警
    assert capsys.readouterr().err == ""
