"""模型延迟档案 + 延迟选路测试（问题5）：全部离线，无需 API key / 网络。

覆盖：
- LatencyStore：取样本中位数（抗单次长尾）、非法值忽略、落盘后换实例可读回、
  按档案路径进程内共享（防多池互相覆写丢更新）、并发落盘不丢样本
- LatencyProber：从未探测过则首次即到期、一轮未结束不叠新一轮（单飞）、
  探测覆盖**全部节点（含冷却/隔离中的不可用节点）**、
  探测失败不写样本也**不**改可用性判定（不进冷却、不当日隔离）
- pick()：按延迟升序、只留延迟最低的 60%（>5 个时等于弃掉最高延迟的 40%）、
  小池上同一公式落在"最低的 2~3 个"、证据不足（冷启动）不裁剪、显式 limit 不裁剪
- 真实成功派工回填延迟档案；同站回退优先挑延迟更低的候选
"""
from __future__ import annotations

import threading
import time

import agent.core.orchestrator as orch
from agent.config import Config, _interpolate
from agent.core.latency import LatencyProber, LatencyStore, get_latency
from agent.core.llm import LLMError
from agent.core.orchestrator import WorkerPool


def _lat_cfg(models, **collab) -> Config:
    return Config(_interpolate({
        "agents": [{"name": "st", "base_url": "u", "api_key": "k", "models": models}],
        "collaboration": collab,
    }))


def _mk_pool(tmp_path, models, latencies=None, tag="lat", **collab) -> WorkerPool:
    """建一个带**独立**延迟档案的池，并把给定延迟灌进档案（None=空档案/冷启动）。"""
    store = LatencyStore(tmp_path / f"{tag}.json")
    for name, ms in (latencies or {}).items():
        store.record(name, ms)
    return WorkerPool(_lat_cfg(models, **collab), exclude=None, latency=store)


def _workers(models: list[str]) -> list[str]:
    """模型名 → 池内工人名（单模型站的工人名格式 st:<model>）。"""
    return [f"st:{m}" for m in models]


# ---------------- 档案层 ----------------

def test_latency_store_median_and_persistence(tmp_path):
    """取最近样本中位数（单次长尾不改变结论）；非法值忽略；换实例能从盘上读回。"""
    path = tmp_path / "lat.json"
    s = LatencyStore(path)
    for ms in (100.0, 120.0, 5000.0):
        s.record("w1", ms)
    assert s.get_ms("w1") == 120.0, "中位数抗单次长尾"
    assert s.get_ms("w2") is None, "无样本返回 None（不是 0）"
    s.record("w1", "not-a-number")  # 非法值忽略，不抛
    s.record("w1", -5)              # 负数忽略
    assert s.get_ms("w1") == 120.0
    assert LatencyStore(path).snapshot() == {"w1": 120.0}, "换实例应能从盘上读回"


def test_latency_store_keeps_only_recent_samples(tmp_path):
    """每工人只留最近 5 个样本：老样本被挤掉，结论跟着最新表现走。"""
    s = LatencyStore(tmp_path / "lat.json")
    for ms in (10.0, 20.0, 30.0, 40.0, 50.0, 900.0):
        s.record("w1", ms)
    assert s.get_ms("w1") == 40.0, "window 内是 [20,30,40,50,900] → 中位数 40"


def test_latency_store_concurrent_records_all_persisted(tmp_path):
    """并发 record 不得互相覆写：落盘的档案要含全部工人样本、不留临时文件。

    与 health 同一口径：快照必须与写盘同在 _write_lock 内取，分开取则写盘顺序可能与
    快照顺序相反，旧快照（少样本）覆盖新快照。这条在 Windows 本地很难复现，
    CI ubuntu 上 health 的同款写法实测丢 2/100 条。
    """
    import json

    path = tmp_path / "lat.json"
    s = LatencyStore(path)

    def writer(idx: int) -> None:
        for j in range(20):
            s.record(f"lt:{idx}-{j}", 10.0 + j)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["models"]) == 100
    assert not list(tmp_path.glob("*.tmp")), "失败路径要清掉临时文件"


def test_get_latency_shares_instance_per_path(tmp_path):
    """同一档案路径全进程共享一个账本；直连构造保留独立实例语义。"""
    path = tmp_path / "shared.json"
    a, b = get_latency(path), get_latency(path)
    assert a is b, "同路径必须共享（否则多池各持账本互相覆写、看不到对方的新鲜样本）"
    a.record("w1", 10.0)
    assert b.get_ms("w1") == 10.0
    assert get_latency(tmp_path / "other.json") is not a, "不同路径不同实例"
    assert LatencyStore(path) is not a, "直连构造保留独立实例（模拟新进程重读盘）"


# ---------------- 探测器 ----------------

def test_probe_round_covers_all_nodes_including_unavailable(tmp_path):
    """探测必须覆盖**全部节点**（含冷却中 / 当日隔离的），且不改可用性判定。

    "包括不可用节点"是用户明确要求：坏节点恢复后不能没有延迟基线，否则一恢复就被
    当成"未测过"的未知节点，选路无从判断。而探测失败只丢样本，绝不制造冷却/隔离——
    否则定时探测器会变成自己给自己制造隔离的噪声源。
    """
    pool = _mk_pool(tmp_path, ["m1", "m2", "m3"])
    pool._cooldowns["st:m2"] = time.time() + 999          # 冷却中
    pool.health.record_failure("st:m3", "m3", "")         # 当日隔离
    seen: list[str] = []

    def fake_probe(name):
        seen.append(name)
        return None if name == "st:m2" else 42.0          # m2 探测失败

    prober = LatencyProber(pool.latency, 600, fake_probe)
    res = prober.probe_all(pool.names())

    assert sorted(seen) == ["st:m1", "st:m2", "st:m3"], "必须覆盖全部节点，含不可用节点"
    assert res["st:m1"] == 42.0 and res["st:m2"] is None
    assert pool.latency.get_ms("st:m1") == 42.0, "探测成功应写进档案"
    assert pool.latency.get_ms("st:m2") is None, "探测失败不写样本"
    assert pool._skip_reason("st:m2") == "冷却中", "探测不得改变冷却状态"
    assert pool._skip_reason("st:m3") == "当日失败隔离", "探测不得改变隔离状态"
    assert pool._fail_streak.get("st:m2") is None, "探测失败不得累计冷却连击"
    assert prober.stats()["rounds"] == 1


def test_prober_is_lazy_and_single_flight(tmp_path):
    """懒式定时：从未探测过 ⇒ 首次即到期；一轮在飞不叠新一轮；探完按周期再到期。"""
    pool = _mk_pool(tmp_path, ["m1", "m2"])
    clock = [1000.0]
    started, release = threading.Event(), threading.Event()

    def slow_probe(name):
        started.set()
        release.wait(5)
        return 7.0

    prober = LatencyProber(pool.latency, 600, slow_probe, clock=lambda: clock[0])
    assert prober.due() is True, "从未探测过 ⇒ 首次检查即到期（冷启动尽快拿基线）"
    assert prober.maybe_start(["st:m1"]) is True
    assert started.wait(5), "应真的起了后台探测线程"
    assert prober.maybe_start(["st:m1"]) is False, "一轮未结束不叠新一轮（单飞）"
    assert prober.due() is False, "在飞期间不到期"
    release.set()
    for _ in range(100):
        if prober.stats()["rounds"] == 1:
            break
        time.sleep(0.05)
    assert prober.stats()["rounds"] == 1 and prober.stats()["running"] is False
    assert prober.due() is False, "刚探完、未到周期 ⇒ 不再到期"
    clock[0] += 601
    assert prober.due() is True, "过了周期 ⇒ 到期"


def test_touch_latency_probe_covers_unavailable_nodes(tmp_path, monkeypatch):
    """派工入口触发的探测同样带上全部工人（含冷却中的）——这是"每 10 分钟探测所有节点"的落点。"""
    monkeypatch.setenv("MAO_LATENCY_PROBE", "1")  # 本用例单独开启探测（其余用例由夹具关闭）
    pool = WorkerPool(_lat_cfg(["m1", "m2"]), exclude=None,
                      latency=LatencyStore(tmp_path / "l.json"))
    assert pool._prober is not None, "latency_probe=true 时应有探测器"
    pool._cooldowns["st:m2"] = time.time() + 999
    seen: list[str] = []
    pool._prober._probe_one = lambda name: (seen.append(name), 5.0)[1]

    pool.pick()  # 派工入口顺带触发一轮后台探测
    for _ in range(100):
        if len(seen) >= 2:
            break
        time.sleep(0.05)
    assert sorted(seen) == ["st:m1", "st:m2"], f"探测应覆盖全部节点，实际 {sorted(seen)}"
    assert pool.latency.get_ms("st:m1") == 5.0


def test_probe_disabled_by_config_does_not_probe(tmp_path):
    """latency_probe 关（本仓测试夹具默认关）⇒ 派工绝不自动探测：离线用例零网络。"""
    pool = _mk_pool(tmp_path, ["m1", "m2"])
    assert pool._latency_auto_probe is False
    seen: list[str] = []
    pool._prober._probe_one = lambda name: (seen.append(name), 5.0)[1]
    pool.pick()
    time.sleep(0.2)  # 给后台线程一点时间：若真起了探测，这里会看到
    assert seen == [], "关闭自动探测时不得起探测（否则离线用例会真打网络）"
    assert pool.latency.snapshot() == {}, "关闭自动探测 ⇒ 档案不被写入"


def test_probe_latency_now_forces_round_even_when_auto_off(tmp_path):
    """手动 probe_latency_now 不受自动开关影响：随时可强制刷一轮全部节点。"""
    pool = _mk_pool(tmp_path, ["m1", "m2"])
    pool._cooldowns["st:m2"] = time.time() + 999  # 不可用节点也要被探（恢复后需要基线）
    seen: list[str] = []
    pool._prober._probe_one = lambda name: (seen.append(name), 33.0)[1]
    res = pool.probe_latency_now()
    assert sorted(seen) == ["st:m1", "st:m2"], "手动探测同样覆盖全部节点（含不可用节点）"
    assert res == {"st:m1": 33.0, "st:m2": 33.0}
    assert pool.latency_status()["prober"]["rounds"] == 1


def test_probe_uses_short_timeout_and_no_retry(tmp_path, monkeypatch):
    """探测必须走**独立短超时 + 不重试**的客户端，不能沿用真实派工的 llm.timeout×重试。

    否则一个卡住的坏节点会把一次"极短测速"拖成分钟级、白打多次请求，同步 probe
    （bridge/MCP `latency probe:true`）更会挂住整轮——正是本次修复要根治的隐患。
    """
    captured: list[dict] = []

    class Recorder:
        def __init__(self, base_url, api_key, model, **kw):
            captured.append({"model": model, **kw})

        def chat(self, messages):
            return {"role": "assistant", "content": "pong"}

        def close(self):
            pass

    monkeypatch.setattr(orch, "LLMClient", Recorder)
    pool = _mk_pool(tmp_path, ["m1", "m2"])

    ms = pool._probe_one("st:m1")
    assert ms is not None, "有效回复应返回耗时（None 只代表空回复被丢弃）"
    assert len(captured) == 1
    assert captured[0]["max_retries"] == 0, "探测客户端不得重试（只发一次请求）"
    assert captured[0]["timeout"] == pool._latency_probe_timeout, "探测用短超时，不沿用派工 llm.timeout"
    assert captured[0]["timeout"] < 60, "探测超时必须显著短于派工超时（默认 8s vs 120s）"

    pool._probe_one("st:m1")  # 同一工人第二次探测应复用缓存客户端，不再重建连接池
    assert len(captured) == 1, "探测客户端应按工人缓存复用"
    assert pool._clients == {}, "探测绝不写入真实派工的 _clients 缓存（两套池互不影响）"


def test_bridge_latency_command_reports_and_refreshes(tmp_path, monkeypatch):
    """bridge `latency` 指令：回延迟档案 + 探测状态；probe=true 时同步刷一轮再回包。"""
    from agent import bridge as br

    monkeypatch.setenv("MAO_LATENCY_PROBE", "1")
    cfg = _lat_cfg(["m1", "m2"])
    b = br.Bridge(cfg)
    try:
        seen: list[str] = []
        b.workers._prober._probe_one = lambda name: (seen.append(name), 33.0)[1]
        out = b.handle({"cmd": "latency", "probe": True})
        assert out["ok"] is True
        assert sorted(seen) == ["st:m1", "st:m2"]
        assert out["model_latency"] == {"st:m1": 33.0, "st:m2": 33.0}
        assert out["auto_probe"] is True and out["prober"]["rounds"] == 1
        # health 也带上延迟字段：外部主一次预检就能同时看到"能不能用"和"快不快用"
        hs = {x["worker"]: x for x in b.handle({"cmd": "health"})["workers"]}
        assert hs["st:m1"]["latency_ms"] == 33.0 and hs["st:m1"]["available"] is True
    finally:
        b.close()


# ---------------- 延迟选路（pick） ----------------

def test_pick_drops_slowest_share_and_sorts_by_latency(tmp_path):
    """10 个可用：只用延迟最低的 60%（6 个），最高延迟的 40% 永不被选中；候选按延迟升序。"""
    models = [f"m{i}" for i in range(1, 11)]
    lat = {f"st:m{i}": i * 10.0 for i in range(1, 11)}
    pool = _mk_pool(tmp_path, models, lat, max_participants=5)

    assert pool.pick() == ["st:m1", "st:m2", "st:m3", "st:m4", "st:m5"], \
        "先按延迟升序排序，再取人数上限（优先派延迟低的）"
    got: set[str] = set()
    for _ in range(8):
        got |= set(pool.pick())
    assert got <= {f"st:m{i}" for i in range(1, 7)}, f"最高延迟的 40% 永不该派工：{sorted(got)}"
    assert {"st:m1", "st:m6"} <= got, f"保留的 60% 应被轮转覆盖到：{sorted(got)}"


def test_pick_small_pool_keeps_two_or_three(tmp_path):
    """低于 5 个可用时，同一公式落在"延迟最低的 2~3 个"（用户口径，不另设分支）。"""
    lat4 = {f"st:m{i}": i * 10.0 for i in range(1, 5)}  # n=4 → 留 3
    pool4 = _mk_pool(tmp_path, ["m1", "m2", "m3", "m4"], lat4, tag="p4")
    got4 = {w for _ in range(4) for w in pool4.pick()}
    assert got4 == {"st:m1", "st:m2", "st:m3"}, f"4 个可用应用最低的 3 个：{sorted(got4)}"

    lat3 = {f"st:m{i}": i * 10.0 for i in range(1, 4)}  # n=3 → 留 2
    pool3 = _mk_pool(tmp_path, ["m1", "m2", "m3"], lat3, tag="p3")
    got3 = {w for _ in range(4) for w in pool3.pick()}
    assert got3 == {"st:m1", "st:m2"}, f"3 个可用应用最低的 2 个：{sorted(got3)}"

    lat2 = {f"st:m{i}": i * 10.0 for i in range(1, 3)}  # n=2 → 留 2（不可再裁）
    pool2 = _mk_pool(tmp_path, ["m1", "m2"], lat2, tag="p2")
    got2 = {w for _ in range(3) for w in pool2.pick()}
    assert got2 == {"st:m1", "st:m2"}, f"2 个可用全留：{sorted(got2)}"


def test_pick_without_enough_latency_data_does_not_prune(tmp_path):
    """证据不足不动刀：空档案（冷启动）不裁剪；样本数不到下限不裁剪；显式 limit 不裁剪。"""
    models = ["m1", "m2", "m3", "m4"]
    cold = _mk_pool(tmp_path, models, tag="cold")
    assert cold.pick() == ["st:m1", "st:m2", "st:m3", "st:m4"], \
        "无任何延迟样本 ⇒ 原样返回（旧行为不变，冷启动不把工人踢出去）"

    one = _mk_pool(tmp_path, models, {"st:m1": 10.0}, tag="one")
    assert len(one.pick()) == 4, "已知样本少于 latency_min_samples(2) ⇒ 不裁剪"

    two = _mk_pool(tmp_path, models, {"st:m1": 10.0, "st:m2": 20.0}, tag="two")
    assert len(two.pick()) == 3, "证据够了才按 60% 裁（4 → 3）"
    # 显式 limit（含 0=不限）表示调用方自己定规模，不做延迟裁剪（只受轮转游标影响）
    assert sorted(two.pick(limit=4)) == _workers(models)
    assert sorted(two.pick(limit=0)) == _workers(models)


# ---------------- 与派工/回退链的联动 ----------------

def test_successful_dispatch_backfills_latency(tmp_path, monkeypatch):
    """真实成功派工把往返耗时写回同一份档案（探测给基线，真实负载给实际表现）。"""
    class Slowish:
        def __init__(self, base_url, api_key, model, **kw):
            self.model = model

        def chat(self, messages):
            time.sleep(0.02)
            return {"role": "assistant", "content": "ok"}

    monkeypatch.setattr(orch, "LLMClient", Slowish)
    pool = _mk_pool(tmp_path, ["m1", "m2"])  # 单模型站会塌成站名，用两个模型保持 st:<model> 命名
    r = pool.ask_result("st:m1", "任务")
    assert r["ok"] is True and r["answer"] == "ok"
    ms = pool.latency.get_ms("st:m1")
    assert ms is not None and ms >= 20.0, f"真实成功往返应回填延迟档案，实际 {ms}"
    assert pool.latency.get_ms("st:m2") is None, "没派工的工人不该凭空有样本"


def test_fallback_prefers_low_latency_peer(tmp_path, monkeypatch):
    """回退候选也按延迟升序挑：同站两个候选中优先选延迟更低的那个。"""
    class M1Boom:
        def __init__(self, base_url, api_key, model, **kw):
            self.model = model

        def chat(self, messages):
            if self.model == "m1":
                raise LLMError("rate_limit", "429", retryable=True)  # 仅 m1 瞬时失败
            return {"role": "assistant", "content": f"ok-from-{self.model}"}

    monkeypatch.setattr(orch, "LLMClient", M1Boom)
    pool = _mk_pool(tmp_path, ["m1", "m2", "m3"], {"st:m2": 900.0, "st:m3": 10.0})
    r = pool.ask_result("st:m1", "任务")
    assert r["worker"] == "st:m3", f"应回退到延迟更低的同站候选，实际 {r['worker']}"
    assert r["answer"] == "ok-from-m3"
