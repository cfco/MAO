"""第八轮审计（2026-10-29）P1/P2 回归：状态一致性、契约与外壳出口。

安全类（shell 包装绕过、密钥外发、stdout 掩码）在 `test_audit8_security.py`，本文件收其余：

1. 自动下线的 `disabled` 标记只能在 registry 真的写上 `#` 之后落——写失败就落标记，
   `if not entry.get("disabled")` 会让这条模型**永远不再重试下线**。
2. 健康档案落盘要能在"读盘之后、写盘之前"被对端进程插队时重来一轮（跨进程 lost update，
   同仓 `model_registry.retire_model` 早就用写前重读封口，这是第三处）。
3. 档案写不进去不再一声不吭（Windows 上 `os.replace` 撞目标被占用是常态）。
4. 发现器不再把别的站的免费清单叠进本站（适配器写死网关 URL），也不再误把付费条目当"下架"；
   不可信远端给的脏 id 进不了叠加层。
5. 热同步整本换掉工人池时，回退选路不得打穿"永不抛异常"的派工契约。
6. 投票平票由候选编号定序，不再由线程完成先后决定；`select_best` 的阈值真的参与判定。
7. 空正文不进候选池；`workers: []` 不再被折叠成"全池派工"。
8. 三外壳的工具/技能出口统一脱敏；`mao call` 崩了也回一行 JSON；MCP 外壳单条失败不炸会话。
9. 配置的 `${VAR}` 解析为空时按站+字段点名；`discovery` 与 note/tags 同一插值口径；
   `.env` 里的小写变量名在 Linux 上不再静默失效。
10. 探测客户端构造失败不得吃掉一格站门；`stop_all` 要等后台线程收完（不留孤儿子进程）。

全部用例离线、无网络。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from agent import config_util as cu
from agent.config import Config, _interpolate
from agent.core import health as health_mod
from agent.core import ledger as ledger_mod
from agent.core import orchestrator as orch
from agent.core.health import ModelHealth
from agent.core.node_discovery import (
    Adapter,
    FreeModel,
    FreeNodeDiscoverer,
    _clean_models,
    _reconcile,
)
from agent.core.orchestrator import WorkerPool
from agent.model_registry import RETIRE_MISSING
from agent.tools.base import FunctionTool, ToolResult

REGISTRY_TXT = "NODE_A_MODELS=m-one@256k,paid-pro@256k\nNODE_B_MODELS=b1:free@128k\n"


def _day(offset: int) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(time.time() + offset * 86400))


def _registry(tmp_path) -> Path:
    reg = tmp_path / "model_registry.txt"
    reg.write_text(REGISTRY_TXT, encoding="utf-8")
    return reg


def _pool(**collab) -> WorkerPool:
    cfg = Config(
        _interpolate(
            {
                "agents": [
                    {"name": "st", "base_url": "u1", "api_key": "k", "models": ["m1", "m2"]},
                    {"name": "other", "base_url": "u2", "api_key": "k2", "model": "x1"},
                ],
                "collaboration": collab or None,
            }
        )
    )
    return WorkerPool(cfg, exclude=None)


# ---------------- 1. 自动下线：标记只能跟着写入成功走 ----------------


def test_disabled_flag_is_only_set_when_registry_write_landed(tmp_path, monkeypatch):
    """写 registry 失败时不得落 `disabled`，否则这条模型永久失去下线机会。"""
    real = health_mod.retire_model
    calls = []

    def flaky(path, env_var, model):
        calls.append(1)
        if len(calls) == 1:
            return RETIRE_MISSING  # 第一次：文件被占用/并发写者打架耗尽重试
        return real(path, env_var, model)

    monkeypatch.setattr(health_mod, "retire_model", flaky)
    reg = _registry(tmp_path)
    h = ModelHealth(tmp_path / "h.json", reg, retire_days=2)
    h.record_failure("st:m-one", "m-one", "NODE_A_MODELS", day=_day(-2))
    h.record_failure("st:m-one", "m-one", "NODE_A_MODELS", day=_day(-1))
    assert "disabled" not in h.snapshot()["st:m-one"], (
        "下线写入失败却落了 disabled → `if not entry.get('disabled')` 让它再也不重试"
    )
    assert "#m-one" not in reg.read_text(encoding="utf-8")

    # 占用释放后，下一个失败日仍要完成下线（旧写法在这里已经永久放弃了）
    h.record_failure("st:m-one", "m-one", "NODE_A_MODELS", day=_day(0))
    assert "#m-one" in reg.read_text(encoding="utf-8"), "写入恢复后必须补上下线"
    assert h.snapshot()["st:m-one"].get("disabled") == _day(0)


# ---------------- 2/3. 公共落盘层：跨进程 CAS 与失败可见 ----------------


def test_write_remerges_when_peer_wrote_after_our_read(tmp_path):
    """读盘 → 写盘之间被对端进程插队时整轮重来，绝不拿过时快照覆盖对方记录。

    旧写法只在 `write()` 开头读一次盘：a 读完、b 写完落盘、a 再整篇覆盖 ⇒ b 的记录没了。
    这里把"对端写入"精确塞进 a 取快照的那一刻，正好命中那个窗口。
    """
    store = tmp_path / "h.json"
    a = ModelHealth(store, None, 7)
    b = ModelHealth(store, None, 7)
    assert not a.quarantined("w-a") and not b.quarantined("w-b")  # 双方各自完成首次读盘

    original = a._snapshot_locked
    fired = []

    def snapshot_after_peer_write():
        if not fired:
            fired.append(1)
            ledger_mod.atomic_write_text(
                store,
                json.dumps(
                    {
                        "version": 2,
                        "active_dates": [],
                        "models": {"w-b": {"fail_dates": [_day(0)]}},
                    }
                ),
            )
        return original()

    a._snapshot_locked = snapshot_after_peer_write  # 只改这个实例，影响面最小
    a.record_failure("w-a", "m-a", "")

    keys = set(json.loads(store.read_text(encoding="utf-8")).get("models", {}))
    assert {"w-a", "w-b"} <= keys, f"后写者不得覆盖对端记录：{sorted(keys)}"


def test_write_failure_is_reported_once_on_stderr(tmp_path, monkeypatch, capsys):
    """落盘失败必须可见：静默 pass 等于档案从此不再更新而没人知道。"""
    ledger_mod._write_warned.clear()

    def boom(path, text):
        raise OSError(13, "Permission denied by scanner")

    monkeypatch.setattr(ledger_mod, "atomic_write_text", boom)
    h = ModelHealth(tmp_path / "h.json", None, 7)
    h.record_failure("w-1", "m1", "")
    h.record_failure("w-2", "m2", "")
    err = capsys.readouterr().err
    assert err.count("[档案]") == 1, f"同一档案只点名一次：\n{err}"
    assert "h.json" in err and "Permission denied" in err


def test_non_string_fail_dates_are_dropped_on_load(tmp_path):
    """档案里混进整数日期不得把记账打成 TypeError。

    `sorted(set(fail_dates) | {day})` 在 str/int 混排时抛 TypeError，而调用方
    （orchestrator._record_result）整段吞异常 ⇒ 当日隔离与连击下线**静默失效**，
    表现是"坏模型当天还在被反复派工"，比崩掉更难查。
    """
    store = tmp_path / "h.json"
    store.write_text(
        json.dumps({"version": 2, "active_dates": [], "models": {"w": {"fail_dates": [20260101]}}}),
        encoding="utf-8",
    )
    h = ModelHealth(store, None, 7)
    h.record_failure("w", "m", "", day=_day(0))  # 载入时整数日期已被过滤，这里只应记到今天
    assert h.quarantined("w"), "混类型条目不得让当日隔离失效"
    assert h.snapshot()["w"]["fail_dates"] == [_day(0)]


def test_health_model_keys_are_capped(tmp_path, monkeypatch):
    """键数封顶：自动发现天天造新工人名，档案不能只增不减（延迟侧本来就有 _trim）。"""
    monkeypatch.setattr(health_mod, "_MAX_MODEL_KEYS", 8)
    h = ModelHealth(tmp_path / "h.json", None, 7)
    for i in range(30):
        # 失败日逐个往后推：裁剪按"最近一次失败日"排序，同一天会退化成按名字比
        h.record_failure(f"w-{i}", f"m-{i}", "", day=_day(i - 30))
    # 断言落在**档案文件**上：那才是"只增不减"的泄漏面（内存视图允许短暂超前）
    stored = json.loads((tmp_path / "h.json").read_text(encoding="utf-8"))["models"]
    assert len(stored) <= 8, f"落盘的账本没被裁剪：{len(stored)} 条"
    assert "w-29" in stored, "裁剪应留下最近失败的条目"
    assert "w-0" not in stored, "最旧的条目才该被裁掉"


# ---------------- 4. 发现器：站归属、付费条目、远端脏 id ----------------


class _SpyAdapter(Adapter):
    name = "kilo"
    host = "api.kilo.ai"
    free_marker = ":free"

    def __init__(self, live):
        self.live = live
        self.calls = 0

    def fetch(self, base_url, api_key, timeout):
        self.calls += 1
        return list(self.live)


def _discoverer(tmp_path, base_url, adapter):
    registry = _registry(tmp_path)
    cfg = Config(
        _interpolate(
            {
                "agents": [
                    {
                        "name": "node-a",
                        "base_url": base_url,
                        "api_key": "k",
                        "models": ["m-one", "paid-pro"],
                        "discovery": "kilo",
                    }
                ],
                "collaboration": {
                    "free_node_discovery": True,
                    "free_node_interval": 0,
                    "free_node_timeout": 2,
                },
            }
        )
    )
    for p in cfg.agent_profiles:
        p.models_env = "NODE_A_MODELS"
    overlay = tmp_path / "free_nodes.json"
    # 叠加层/清单路径都指到 tmp，绝不碰仓库里的真文件
    return FreeNodeDiscoverer(cfg, overlay, adapters={"kilo": adapter}), overlay, registry


def test_discovery_skips_station_that_does_not_belong_to_adapter(tmp_path):
    """适配器写死自己的公开网关时必须校验本站归属，否则把别人的清单叠进本站 env。"""
    adapter = _SpyAdapter([FreeModel("m-one:free", 128)])
    d, overlay, _reg = _discoverer(tmp_path, "https://router.other-station.id/v1", adapter)
    report = d.run_once()
    assert adapter.calls == 0, "站不属于该适配器却仍然去拉清单"
    assert report["stations"]["NODE_A_MODELS"]["reason"] == "host_mismatch"
    assert not overlay.exists(), "不该写叠加层"


def test_paid_entries_are_not_retired_by_a_free_only_list():
    """免费清单里没有 ≠ 已下架：人工加的付费模型不能被发现轮摘掉。"""
    adapter = _SpyAdapter([])
    delta = _reconcile("b1:free@128k,paid-pro@256k", [FreeModel("b2:free")], adapter=adapter)
    assert delta["retired"] == ["b1:free"], f"retired 只应作用于免费档条目：{delta['retired']}"


def test_dirty_remote_ids_never_reach_the_overlay(tmp_path):
    """含逗号/空白的远端 id 进叠加层 = 往 `NODE_*_MODELS` 注入多条模型。"""
    kept = _clean_models([FreeModel("good-id"), FreeModel("a,b"), FreeModel("has space")])
    assert [m.model for m in kept] == ["good-id"]

    adapter = _SpyAdapter([FreeModel("a,b")])
    d, overlay, _reg = _discoverer(tmp_path, "https://api.kilo.ai/api/gateway", adapter)
    report = d.run_once()
    assert report["stations"]["NODE_A_MODELS"]["reason"] == "no_valid_models"
    assert not overlay.exists()


# ---------------- 5. 热同步换池与派工契约 ----------------


class _SwappingDict(dict):
    """`in` 判定通过后就把池子换成"不含该键"的新视图（模拟 sync_free_nodes 整本替换）。"""

    def __init__(self, target, source):
        super().__init__(source)
        self._target = target
        self._armed = False

    def __contains__(self, key):
        self._armed = True
        return super().__contains__(key)

    def __getitem__(self, key):
        if self._armed and key not in self._target:
            raise KeyError(key)
        return super().__getitem__(key)

    def items(self):
        return {}.items() if self._armed else super().items()


def test_fallback_target_survives_profiles_being_swapped(monkeypatch):
    """池子在"判定存在"与"取实例"之间被整本换掉时，回退选路不得抛出。"""
    pool = _pool(fallback_cross_station=True)
    names = list(pool.profiles)
    lead = pool.profiles["st:m1"]
    swapped = {n: pool.profiles[n] for n in names if n != "st:m2"}
    monkeypatch.setattr(pool, "profiles", _SwappingDict(swapped, pool.profiles))

    # st 站的成员表里仍有 st:m2 ⇒ 旧写法 `n in self.profiles` 过关、`self.profiles[name]` 炸
    result = pool._fallback_target(lead, {"st:m1"})
    assert result is None or result.name in names, f"应正常返回，实际 {result}"


# ---------------- 6. 投票：平票定序、阈值真判、空候选短路 ----------------


class _BallotHost:
    """只喂 `_run_ballot` 需要的最小面：完成顺序可控的 _run_parallel。"""

    vote_preview_chars = 240

    class _Cfg:
        vote_preview_chars = 240

    def __init__(self, pairs):
        self._pairs = pairs
        self.cfg = self._Cfg()

    def _effective_timeout(self, n, timeout, workers=None):
        return 1.0

    def _run_parallel(self, workers, prompt, system, timeout):
        return {w: {"status": "ok", "answer": ans} for w, ans in self._pairs}


def test_ballot_tie_is_broken_by_candidate_number_not_completion_order():
    """同一批选票、两种完成顺序，必须选出同一个胜者（重跑不得看到答案互换）。"""
    from agent.core.voting import VotingMixin

    cands = [("a", "AAA"), ("b", "BBB")]
    x = VotingMixin._run_ballot(_BallotHost([("w1", "1"), ("w2", "2")]), cands, ["w1", "w2"], None)
    y = VotingMixin._run_ballot(_BallotHost([("w2", "2"), ("w1", "1")]), cands, ["w1", "w2"], None)
    assert x and y
    assert x[0] == y[0] == 1, f"平票应固定由编号小者胜出：{x[0]} vs {y[0]}"


def test_select_best_honours_threshold_and_skips_empty_candidates(monkeypatch):
    pool = _pool()
    calls = []
    monkeypatch.setattr(pool, "_run_parallel", lambda *a, **k: calls.append(1) or {})

    assert pool.select_best([], voters=pool.names()) is None
    assert not calls, "空候选仍发起了投票轮（白烧一轮额度）"

    answers = {"1": "1", "2": "2", "3": "3"}
    names = list(pool.profiles)

    def fake_parallel(workers, prompt, system, timeout):
        # 三个投票人各投一张，票型 1/1/1 —— 任何阈值高于 1/3 都不该算"择优成功"
        return {
            w: {"status": "ok", "answer": answers[str(i % 3 + 1)]} for i, w in enumerate(workers)
        }

    monkeypatch.setattr(pool, "_run_parallel", fake_parallel)
    cands = [("a", "AAA"), ("b", "BBB"), ("c", "CCC")]
    assert pool.select_best(cands, voters=names, threshold=0.6) is None, "未过阈值不得当胜出"
    picked = pool.select_best(cands, voters=names, threshold=0.3)
    assert picked is not None and picked[0] == "a", picked


# ---------------- 7. 空正文与显式空名单 ----------------


def test_empty_answer_stays_out_of_vote_candidates(monkeypatch):
    """空正文不进候选池：旧写法把它换成 "(空回复)" 哨兵，于是照样参选、还能被投中。"""
    pool = _pool()

    class EmptyClient:
        def chat_once(self, messages, max_tokens=None):
            return {"role": "assistant", "content": ""}

    monkeypatch.setattr(orch, "LLMClient", lambda *a, **k: EmptyClient())
    pool._clients.clear()
    who = pool.names()[0]

    r = pool.ask_result(who, "任务")
    assert r["status"] == "ok" and r["answer"] == "", r
    assert pool.collect(list(pool.profiles), "任务") == [], "空正文不该成为投票候选"
    assert "空" in pool.ask(who, "任务"), "人读文案仍要给出空提示"


def test_explicit_empty_workers_errors_instead_of_pool_wide_dispatch():
    from agent.bridge import Bridge

    bridge = Bridge(Config({"agents": []}))
    try:
        for cmd in ("ask_many", "ask_vote", "run_review"):
            out = bridge.handle({"cmd": cmd, "prompt": "x", "draft": "y", "workers": []})
            assert out["ok"] is False, f"{cmd} 显式空名单必须报错：{out}"
            assert "workers" in out["error"], out
    finally:
        bridge.close()


# ---------------- 8. 外壳出口统一脱敏 ----------------


def test_tool_and_skill_exits_are_masked_in_bridge():
    from agent.bridge import Bridge

    bridge = Bridge(Config({"agents": []}))
    try:
        bridge.registry.register(
            FunctionTool(
                name="leaky",
                description="测试用：正文里带密钥形态",
                input_schema={},
                func=lambda a: ToolResult("NODE_A_KEY=supersecretvalue status=401"),
            )
        )
        out = bridge.handle({"cmd": "call_tool", "name": "leaky", "args": {}})
        assert out["ok"] is True
        assert "supersecretvalue" not in out["result"], out
        assert "NODE_A_KEY=****" in out["result"], out
    finally:
        bridge.close()


def test_call_mode_still_prints_one_json_line_when_handler_raises(monkeypatch, capsys):
    """`mao call` 的契约是"一行 JSON + 退出码"：异常只能变成 ok:false，不能只留崩栈。"""
    import agent.__main__ as main_mod

    def boom(self, req):
        raise RuntimeError("connection to https://api.x/v1?key=sk-abcd123456 failed")

    monkeypatch.setattr(main_mod, "load_config", lambda *a, **k: Config({"agents": []}))
    # 不能真跑 reconfigure_streams：它会把 pytest 的捕获流改写成 utf-8，后续用例的
    # capsys 就开始乱码（同一进程里互相污染）。本用例只验"崩了也要回一行 JSON"。
    monkeypatch.setattr(main_mod, "reconfigure_streams", lambda: None)
    monkeypatch.setattr("agent.bridge.Bridge.handle", boom)
    monkeypatch.setattr("sys.argv", ["agent", "call", "ping", "{}"])
    with pytest.raises(SystemExit) as exc:
        main_mod.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out.strip().splitlines()
    resp = json.loads(out[-1])
    assert resp["ok"] is False and "RuntimeError" in resp["error"]
    assert "sk-abcd123456" not in resp["error"], "错误文案必须脱敏"


def test_mcp_shell_turns_a_handler_crash_into_ok_false():
    """MCP 外壳过去没有 per-call 兜底：一条指令抛异常会带走整个会话（宿主看到工具集体失联）。"""
    import anyio
    from mcp import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    from agent.mcp_server import build_server

    app, bridge = build_server(Config({"agents": []}))

    def boom(req):
        raise RuntimeError("endpoint https://api.x/v1?key=sk-abcd123456 exploded")

    bridge.handle = boom
    got: dict = {}

    async def go(s):
        r = await s.call_tool("ping", {})
        got.update(json.loads(r.content[0].text))

    async def main():
        async with create_client_server_memory_streams() as (client, server):
            async with anyio.create_task_group() as tg:

                async def _serve():
                    low = app._lowlevel_server  # noqa: SLF001 - SDK 测试惯用入口
                    await low.run(server[0], server[1], low.create_initialization_options())

                tg.start_soon(_serve)
                async with ClientSession(client[0], client[1]) as session:
                    await session.initialize()
                    await go(session)
                tg.cancel_scope.cancel()

    try:
        anyio.run(main)
    finally:
        bridge.close()
    assert got["ok"] is False, got
    assert "sk-abcd123456" not in got["error"], got


# ---------------- 9. 配置口径 ----------------


def test_empty_reference_is_named_by_station_and_field(monkeypatch, capsys):
    """`${VAR}` 解析成空 = 变量名拼错或 .env 留空值 ⇒ 当场按站+字段点名（不回显值）。"""
    monkeypatch.setattr(cu, "_empty_ref_logged", set())
    raw = {
        "agents": [
            {"name": "node-a", "base_url": "${NODE_A_ENDPOINT_TYPo}", "api_key": "sk-literal-ok"}
        ]
    }
    Config(_interpolate(raw), raw=raw)
    err = capsys.readouterr().err
    assert "站 'node-a' 的 base_url" in err, err
    assert "api_key" not in err.split("站 'node-a' 的 base_url")[-1][:60], "写死的 key 不该被点名"


def test_discovery_is_read_after_interpolation():
    """discovery 与 note/tags 必须同一插值口径，否则 `discovery: ${VAR}` 静默永不发现。"""
    os.environ["MAO_TEST_DISCOVERY"] = "kilo"
    try:
        raw = {
            "agents": [
                {
                    "name": "node-b",
                    "base_url": "u",
                    "api_key": "k",
                    "model": "m1",
                    "discovery": "${MAO_TEST_DISCOVERY}",
                }
            ]
        }
        cfg = Config(_interpolate(raw), raw=raw)
        assert cfg.agent_profiles[0].discovery == "kilo", cfg.agent_profiles[0].discovery
    finally:
        os.environ.pop("MAO_TEST_DISCOVERY", None)


def test_lowercase_dotenv_key_gets_upper_alias(tmp_path, monkeypatch, capsys):
    """.env 写小写变量名：Windows 能用、Linux/CI 取不到 ⇒ 注入时补大写别名并点名一次。

    断言必须**平台无关**：第一版直接查 `os.environ["NODE_A_ENDPOINT"]`，在 Windows 上
    即便删掉补别名的那行也照样命中（`os.environ` 键名不区分大小写），变异门禁把这条记成
    TAUTOLOGY。改查注入登记表 `_dotenv_keys`——它是普通 set，不受平台大小写规则影响。
    """
    import agent.config_load as acl

    injected: set[str] = set()
    monkeypatch.setattr(acl, "_dotenv_keys", injected)
    monkeypatch.setattr(acl, "_lowercase_warned", set())
    env_file = tmp_path / ".env"
    env_file.write_text("node_a_endpoint=https://router.example/v1\n", encoding="utf-8")
    saved = os.environ.pop("NODE_A_ENDPOINT", None)
    saved_lower = os.environ.pop("node_a_endpoint", None)
    try:
        acl._load_dotenv(env_file)
        assert "node_a_endpoint" in injected, "原名要照常注入"
        assert "NODE_A_ENDPOINT" in injected, (
            "小写名必须同时登记同名大写：config.yaml 的 ${...} 引用一律是大写，Linux/CI 取不到就是静默失效"
        )
        assert os.environ.get("NODE_A_ENDPOINT") == "https://router.example/v1"
        # 断言只挑 ASCII 片段：控制台捕获的中文在不同平台编码下不可靠
        err = capsys.readouterr().err
        assert "node_a_endpoint" in err and "Linux/CI" in err, err
    finally:
        for k in ("NODE_A_ENDPOINT", "node_a_endpoint"):
            os.environ.pop(k, None)
        if saved is not None:
            os.environ["NODE_A_ENDPOINT"] = saved
        if saved_lower is not None:
            os.environ["node_a_endpoint"] = saved_lower


# ---------------- 10/11. 站门与子进程收尾 ----------------


def test_probe_client_construction_failure_does_not_leak_station_slot(monkeypatch):
    pool = _pool()
    who = pool.names()[0]
    station = pool.profiles[who].station_key
    monkeypatch.setattr(
        pool, "_probe_client_for", lambda p: (_ for _ in ()).throw(RuntimeError("bad ssl"))
    )
    assert pool._probe_one(who) is None
    gate = pool._station_gate(station)
    assert gate.acquire(timeout=0.2), "探测客户端构造失败漏掉一格站门 ⇒ 该站被永久饿死"
    gate.release()


def test_stop_all_waits_for_the_worker_threads_to_finish():
    from agent.tools.mcp_client import stop_all

    done = []

    class FakeConn:
        def __init__(self):
            self._thread = threading.Thread(target=self._run, daemon=True)

        def stop(self):
            pass

        def _run(self):
            threading.Event().wait(0.05)
            done.append(1)

    c = FakeConn()
    c._thread.start()
    stop_all([c])
    assert done, "stop_all 不 join ⇒ 进程随即退出，stdio 子进程留成孤儿"
