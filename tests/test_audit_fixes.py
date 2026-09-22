"""本轮框架审计（2026-09-22）六项修复的回归测试：全部离线。

对应问题（编号见审计报告与 docs/2026-09-22-工作记录.md）：
1. 健康档案多实例 lost update —— 同一档案路径必须全进程共享一个 ModelHealth
   （否则 Web 多会话互相覆盖写盘，且彼此看不到当日隔离）
2. 全池派工规模失控 —— pick() 限制参与人数，_effective_timeout() 按批次自适应
3. MCP 每会话重复建连 —— 进程内共享连接组 + 引用计数
4. 批量路径预过滤绕过缓存 —— 冷却/隔离不再在入口剔除，缓存命中优先
5. 上下文预算漏算 system prompt
6. 计票把错误文本里的数字算成选票

这些用例都是"针对缺陷的断言"（去掉修复即失败），不是恒真用例：
例如 test_hit_cache_still_usable_when_cooling 在恢复预过滤后会立刻挂在 collect 上。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import agent.core.orchestrator as orch
import agent.core.session as sess_mod
import agent.tools.mcp_client as mcp_client
from agent.config import Config, _interpolate
from agent.core.health import ModelHealth, get_health
from agent.core.orchestrator import WorkerPool
from agent.core.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent


def _pool(models: str, cap: int = 3, max_workers: int = 3) -> WorkerPool:
    cfg = Config(_interpolate({
        "agents": [{"name": "st", "base_url": "u", "api_key": "k", "models": models}],
        "collaboration": {"max_participants": cap, "max_workers": max_workers},
    }))
    return WorkerPool(cfg, exclude=None)


# ---------------- 1) 健康档案共享（lost update） ----------------

def test_get_health_returns_same_instance_per_path(tmp_path):
    store = tmp_path / "h.json"
    assert get_health(store) is get_health(store)


def test_shared_health_does_not_lose_each_others_records(tmp_path):
    """原缺陷：h1 记 node-a、h2 再记 node-b，h2 的全量覆写会把 node-a 抹掉。

    必须先让两个实例各自完成"首次读盘"：ModelHealth 是懒加载的，若某一方在
    写盘前才第一次读盘，它会读到对方刚写进去的记录，从而掩盖覆写问题
    （这正是本用例第一版写成恒真用例的原因）。
    """
    store = tmp_path / "h.json"
    h1 = get_health(store)
    h2 = get_health(store)
    h1.quarantined("warmup")  # 触发 h1 首次读盘（此刻是空档案）
    h2.quarantined("warmup")  # 触发 h2 首次读盘（同样是空档案）

    h1.record_failure("node-a", "m-a", "A_MODELS")
    assert h2.quarantined("node-a"), "同进程另一使用者必须立刻看到当日隔离"

    h2.record_failure("node-b", "m-b", "B_MODELS")
    models = json.loads(store.read_text(encoding="utf-8"))["models"]
    assert {"node-a", "node-b"} <= set(models), f"写盘不得互相覆盖：{sorted(models)}"


def test_worker_pools_share_one_health_instance():
    """WorkerPool 是每 Agent 一份（Web 每会话一份），默认健康档案必须共享。"""
    p1 = _pool("m1,m2")
    p2 = _pool("m1,m2")
    assert p1.health is p2.health
    p1.health.record_failure("st:m1", "m1", "")
    assert p2.health.quarantined("st:m1")


def test_direct_construction_stays_independent(tmp_path):
    """直连构造仍返回独立实例：保留"新实例=模拟新进程重读盘"的可测性。"""
    store = tmp_path / "h.json"
    assert ModelHealth(store) is not ModelHealth(store)


# ---------------- 2) 派工规模与限时 ----------------

def test_pick_caps_participants_and_keeps_order():
    pool = _pool("m1,m2,m3,m4,m5,m6,m7,m8", cap=3)
    assert pool.pick() == ["st:m1", "st:m2", "st:m3"], "确定性顺序 + 受上限约束"


def test_pick_skips_unavailable_and_zero_means_unlimited():
    pool = _pool("m1,m2,m3", cap=2)
    pool._cooldowns["st:m2"] = time.time() + 999
    assert pool.pick() == ["st:m1", "st:m3"], "冷却中的工人不占名额"
    assert _pool("m1,m2,m3", cap=0).pick() == ["st:m1", "st:m2", "st:m3"], "0 = 不限"


def test_effective_timeout_scales_with_batches_and_respects_explicit():
    pool = _pool("m1,m2,m3,m4,m5", cap=5, max_workers=3)
    assert pool._effective_timeout(1, None) == 150.0, "1 批 = 120 + 30 余量"
    assert pool._effective_timeout(3, None) == 150.0, "3 名刚好 1 批"
    assert pool._effective_timeout(5, None) == 270.0, "5 名 = 2 批"
    assert pool._effective_timeout(25, None) == 1110.0, "25 名 = 9 批（固定 300s 会截断尾部）"
    assert pool._effective_timeout(25, 0.3) == 0.3, "调用方显式指定优先"


def test_default_fanout_is_bounded(monkeypatch):
    """缺省名单（不传 workers）时，各批量入口的请求数必须被 max_participants 约束。"""
    pool = _pool(",".join(f"m{i}" for i in range(1, 9)), cap=3)
    calls: list[str] = []

    def spy(worker, prompt, system=None):
        calls.append(worker)
        return f"答案[{worker}]"

    monkeypatch.setattr(pool, "ask", spy)

    calls.clear()
    pool.ask_many(pool.pick(), "q")
    assert len(calls) == 3

    calls.clear()
    pool.vote("q")
    assert len(calls) == 6, "收集 3 + 投票 3"

    calls.clear()
    Pipeline(pool).run("q")
    assert len(calls) == 12, "起草 3 + 评审 3 + 修订 3 + 择优投票 3"


def test_explicit_workers_are_not_capped(monkeypatch):
    """调用方显式点名的工人照做（上限只管"缺省名单"的选取）。

    断言"被叫到哪些人"和"返回结果的组织顺序"，不断言"调用的先后"：
    ask_many 是并发派工，各工人线程调 ask 的时序天然抖动（CI 上 m6 就抢在 m5 前）。
    顺序契约在结果层——ask_many 按入参顺序收集输出（见其 docstring），故这里锁它。
    """
    pool = _pool("m1,m2,m3,m4,m5,m6", cap=2)
    calls: list[str] = []

    def spy(worker, prompt, system=None):
        calls.append(worker)
        return f"答案[{worker}]"

    monkeypatch.setattr(pool, "ask", spy)
    out = pool.ask_many(["st:m4", "st:m5", "st:m6"], "q")
    assert sorted(calls) == ["st:m4", "st:m5", "st:m6"], "cap=2 不得截断显式名单"
    assert [b.splitlines()[0] for b in out.split("\n\n")] == [
        "### 工人 st:m4 的结果",
        "### 工人 st:m5 的结果",
        "### 工人 st:m6 的结果",
    ], "输出必须按入参工人顺序排列"


# ---------------- 3) MCP 连接共享 ----------------

def test_mcp_group_shared_and_refcounted():
    servers = [{
        "name": "probe",
        "transport": "stdio",
        "command": "definitely-not-exist-xyz",
        "args": [],
    }]
    key = mcp_client._group_key(servers)
    g1 = mcp_client.acquire_group(servers)
    try:
        g2 = mcp_client.acquire_group(servers)
        assert g1 is g2, "同配置只应有一组连接"
        assert g1.refs == 2
        assert len(g1.conns) == 1, "连接数不随使用者增加"

        mcp_client.release_group(g2)
        assert g1.refs == 1
        assert key in mcp_client._groups, "还有使用者在用，不能关连接"
    finally:
        mcp_client.release_group(g1)
    assert key not in mcp_client._groups, "引用归零后从共享表移除"


# ---------------- 4) 缓存命中优先于冷却（批量路径） ----------------

def test_hit_cache_still_usable_when_cooling():
    """原缺陷：collect/ask_many 在进 ask 之前就按冷却预过滤，连缓存都查不到。"""
    pool = _pool("m1,m2", cap=2)
    worker = pool.names()[0]
    pool._cache[(worker, "已缓存的问题", None)] = "缓存里的答案"
    pool._cooldowns[worker] = time.time() + 999  # 冷却中

    assert pool.ask(worker, "已缓存的问题") == "缓存里的答案"
    assert pool.collect([worker], "已缓存的问题") == [(worker, "缓存里的答案")]
    assert "缓存里的答案" in pool.ask_many([worker], "已缓存的问题")


def test_no_cache_still_reports_skip_in_many():
    pool = _pool("m1,m2", cap=2)
    worker = pool.names()[0]
    pool._cooldowns[worker] = time.time() + 999
    out = pool.ask_many([worker], "没缓存的问题")
    assert "本轮跳过" in out and worker in out


# ---------------- 5) 上下文预算计入 system prompt ----------------

def test_system_prompt_takes_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(sess_mod, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(sess_mod, "TRASH_DIR", tmp_path / "trash")
    s = sess_mod.Session(session_id="budget-probe")
    for _ in range(20):
        s.add("user", "x" * 1000)  # 每条约 250 token

    kept_short = len(s.messages_with("短 system", context_length=4000)) - 1
    kept_long = len(s.messages_with("y" * 3000, context_length=4000)) - 1
    assert kept_long < kept_short, (
        f"长 system 必须挤掉历史（短={kept_short} 长={kept_long}）；"
        "不扣 system 时两者会相同"
    )


# ---------------- 6) 计票不采信错误文本 ----------------

def test_ballot_ignores_error_text(monkeypatch):
    """错误文本里的合法编号（如"错误码 1"）不得被算成一张选票。"""
    pool = _pool("m1,m2", cap=2)
    worker = pool.names()[0]

    class ErrClient:
        def chat(self, messages):
            return {"role": "assistant", "content": "错误：子智能体 'm1' 失败 [api]：错误码 1"}

    monkeypatch.setattr(orch, "LLMClient", lambda *a, **k: ErrClient())
    pool._clients.clear()
    assert pool.select_best([("a", "AAA"), ("b", "BBB")], voters=[worker]) is None


def test_ballot_still_parses_normal_vote(monkeypatch):
    pool = _pool("m1,m2", cap=2)
    worker = pool.names()[0]

    class OkClient:
        def chat(self, messages):
            return {"role": "assistant", "content": "我选第 2 个"}

    monkeypatch.setattr(orch, "LLMClient", lambda *a, **k: OkClient())
    pool._clients.clear()
    picked = pool.select_best([("a", "AAA"), ("b", "BBB")], voters=[worker])
    assert picked is not None and picked[0] == "b"

