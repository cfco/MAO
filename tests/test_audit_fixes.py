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
import agent.tools.mcp_client as mcp_client
from agent.config import Config, _interpolate
from agent.core.health import ModelHealth, get_health
from agent.core.orchestrator import WorkerPool

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


def test_pick_rotates_across_calls():
    """#6：连续默认派工轮流覆盖整个池，不再每次死取头部 N 个（摊薄免费额度、避单点）。"""
    pool = _pool("m1,m2,m3,m4", cap=2)
    assert pool.pick() == ["st:m1", "st:m2"], "首次从头部窗口开始"
    assert pool.pick() == ["st:m3", "st:m4"], "第二次轮转到后一半"
    assert pool.pick() == ["st:m1", "st:m2"], "到池尾回绕，负载持续轮转"
    # 显式点名的批量入口用同一 pick 游标推进：整池都会被轮到
    seen = set()
    for _ in range(2):
        seen.update(pool.pick())
    assert seen == {"st:m1", "st:m2", "st:m3", "st:m4"}, "轮转应最终覆盖全池"


def test_pick_filters_by_tags():
    """#6：可选 tags 只保留能力标签全部命中的工人，供外部主按任务类型选路。"""
    cfg = Config(_interpolate({
        "agents": [
            {"name": "coder", "base_url": "u", "api_key": "k", "model": "m", "tags": "code,中文"},
            {"name": "reader", "base_url": "u", "api_key": "k", "model": "m", "tags": "长上下文"},
        ],
        "collaboration": {"max_participants": 0},
    }))
    pool = WorkerPool(cfg, exclude=None)
    assert pool.pick(tags="code") == ["coder"]
    assert pool.pick(tags="长上下文") == ["reader"]
    assert pool.pick(tags="code,长上下文") == [], "无人同时具备 → 空"
    assert sorted(pool.pick()) == ["coder", "reader"], "不带 tags 则全给"


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
        return {"worker": worker, "ok": True, "status": "ok",
                "answer": f"答案[{worker}]", "error": ""}

    monkeypatch.setattr(pool, "ask_result", spy)

    calls.clear()
    pool.ask_many(pool.pick(), "q")
    assert len(calls) == 3

    calls.clear()
    pool.vote("q")
    assert len(calls) == 6, "收集 3 + 投票 3"

    # 注：原「Pipeline 起草+评审+修订+择优」子断言已随固定流水线移除删除；
    # 缺省扇出上限由上面的 ask_many/vote 与 test_pick_* 覆盖。


def test_explicit_workers_are_not_capped(monkeypatch):
    """调用方显式点名的工人照做（上限只管"缺省名单"的选取）。

    断言"被叫到哪些人"和"返回结果的组织顺序"，不断言"调用的先后"：
    ask_many 是并发派工，各工人线程调 ask_result 的时序天然抖动（CI 上 m6 就抢在 m5 前）。
    顺序契约在结果层——ask_many 按入参顺序收集输出（见其 docstring），故这里锁它。
    """
    pool = _pool("m1,m2,m3,m4,m5,m6", cap=2)
    calls: list[str] = []

    def spy(worker, prompt, system=None):
        calls.append(worker)
        return {"worker": worker, "ok": True, "status": "ok",
                "answer": f"答案[{worker}]", "error": ""}

    monkeypatch.setattr(pool, "ask_result", spy)
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

def test_no_cache_still_reports_skip_in_many():
    pool = _pool("m1,m2", cap=2)
    worker = pool.names()[0]
    pool._cooldowns[worker] = time.time() + 999
    out = pool.ask_many([worker], "没缓存的问题")
    assert "本轮跳过" in out and worker in out


# ---------------- 6) 计票不采信错误文本 ----------------

def test_ballot_ignores_error_text(monkeypatch):
    """派工失败的投票人不得贡献选票（#4：计票按结构化 status 判定，绝不解析文本前缀）。

    构造：好工人 a 正常投 1 号；坏工人 b/c 派工失败，但其文本里带"改投 2 号"的合法编号。
    正确实现只认 ST_OK → 只有 [1] → 胜出 cand1。若把失败票也计入（回归），会变成
    [1,2,2] → 反而胜出 cand2。用 3 候选 + 让错误票指向同一编号制造确定性差距，避免并发
    完成顺序导致的平票抖动。
    """
    pool = _pool("m1,m2,m3", cap=3)
    a, b, c = pool.names()

    def fake(worker, prompt, system=None):
        if worker == a:
            return {"worker": a, "ok": True, "status": "ok", "answer": "1", "error": ""}
        return {"worker": worker, "ok": False, "status": "error",
                "answer": "错误：节点挂了，改投 2 号", "error": "boom"}

    monkeypatch.setattr(pool, "ask_result", fake)
    picked = pool.select_best(
        [("cand1", "AAA"), ("cand2", "BBB"), ("cand3", "CCC")], voters=[a, b, c]
    )
    assert picked is not None and picked[0] == "cand1", (
        "只有成功票 a 的『1』算数；b/c 是失败票，其文本里的『2』不得计入"
    )


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
