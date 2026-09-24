"""同站串行（collaboration.max_per_station）的回归测试：全部离线、不打网络。

策略口径（2026-09-24 定）：**一个中转站同时最多跑一个模型**，避免同站两个模型在站端
互相排队拖慢，并把延迟档案（选路裁剪的依据）污染成"这个模型慢"。

每个用例都是"针对该约束的断言"——去掉站点门（_chat_gated / _probe_one 里的 gate）即失败：
  1) 同站批量派工峰值并发必须为 1；
  2) 跨站必须仍然并行（否则等于把 MAO 退化成串行）；
  3) max_per_station 调大即退回旧的"只看总并发"行为；
  4) 整体限时按"站数 × 每站配额"推导，不能让排队被判成超时丢弃；
  5) 失败回退不得原站在门内递归（反向回退会 A 等 B、B 等 A 死锁）；
  6) 延迟探测遇站被占时让路而不是陪等。
"""
from __future__ import annotations

import threading
import time

import pytest

import agent.core.latency as latency_mod
from agent.config import Config, _interpolate
from agent.core import orchestrator as orch
from agent.core.llm import LLMError
from agent.core.orchestrator import WorkerPool


class _Tracker:
    """按站记录"在飞请求"的峰值并发与总请求数（线程安全）。"""

    def __init__(self, hold: float = 0.05):
        self.hold = hold
        self.lock = threading.Lock()
        self.inflight_station: dict[tuple[str, str], int] = {}
        self.inflight_total = 0
        self.peak_station: dict[tuple[str, str], int] = {}
        self.peak_total = 0
        self.calls: dict[tuple[str, str], int] = {}

    def enter(self, station: tuple[str, str]) -> None:
        with self.lock:
            self.inflight_station[station] = self.inflight_station.get(station, 0) + 1
            self.calls[station] = self.calls.get(station, 0) + 1
            self.peak_station[station] = max(
                self.peak_station.get(station, 0), self.inflight_station[station]
            )
            self.inflight_total += 1
            self.peak_total = max(self.peak_total, self.inflight_total)
        time.sleep(self.hold)

    def leave(self, station: tuple[str, str]) -> None:
        with self.lock:
            self.inflight_station[station] -= 1
            self.inflight_total -= 1


def _pool(station_specs, max_per_station: int = 1, max_workers: int = 3,
          fallback_models: int = 2, cap: int = 0) -> WorkerPool:
    """station_specs: [(站名, base_url, 模型清单串), ...] ⇒ 一站多模型的多站池。"""
    cfg = Config(_interpolate({
        "agents": [
            {"name": nm, "base_url": url, "api_key": f"key-{nm}", "models": models}
            for nm, url, models in station_specs
        ],
        "collaboration": {
            "max_participants": cap, "max_workers": max_workers,
            "max_per_station": max_per_station, "fallback_models": fallback_models,
            "fallback_cross_station": True, "latency_probe": False,
        },
    }))
    return WorkerPool(cfg, exclude=None)


def _fake_client(tracker: _Tracker, fail_once=()):
    """替换 orch.LLMClient：按站记录进出；在 fail_once 里的站首次抛可重试错误。"""

    class FakeClient:
        def __init__(self, base_url, api_key, model, timeout=None, max_retries=None):
            self.station = (base_url, api_key)
            self.model = model

        def chat(self, messages):
            n = tracker.calls.get(self.station, 0) + 1
            if self.station in fail_once and n == 1:
                tracker.enter(self.station)
                tracker.leave(self.station)
                raise LLMError("timeout", "模拟瞬时失败", retryable=True)
            tracker.enter(self.station)
            tracker.leave(self.station)
            return {"role": "assistant", "content": f"ans:{self.model}"}

        def close(self):
            pass

    return FakeClient


# ---------------- 1) 同站串行 ----------------

def test_same_station_batch_never_overlaps(monkeypatch):
    """同站 3 个模型一起派工：站内在飞峰值必须为 1（这就是本策略的全部内容）。"""
    tracker = _Tracker(hold=0.05)
    monkeypatch.setattr(orch, "LLMClient", _fake_client(tracker))
    pool = _pool([("A", "http://a", "m1,m2,m3")])
    try:
        workers = ["A:m1", "A:m2", "A:m3"]
        out = pool.collect(workers, "p")
        assert len(out) == 3, "同站串行后结果一个都不能少"
        assert tracker.peak_station[("http://a", "key-A")] == 1
        assert tracker.calls[("http://a", "key-A")] == 3
    finally:
        pool.close()


# ---------------- 2) 跨站仍然并行 ----------------

def test_cross_station_still_runs_in_parallel(monkeypatch):
    """两站各 2 模型：每站峰值 1，但整体峰值必须到 2（跨站真并行，没退化成串行）。"""
    tracker = _Tracker(hold=0.15)
    monkeypatch.setattr(orch, "LLMClient", _fake_client(tracker))
    pool = _pool([("A", "http://a", "m1,m2"), ("B", "http://b", "m1,m2")])
    try:
        t0 = time.monotonic()
        out = pool.collect(["A:m1", "A:m2", "B:m1", "B:m2"], "p")
        spent = time.monotonic() - t0
        assert len(out) == 4
        assert tracker.peak_station[("http://a", "key-A")] == 1
        assert tracker.peak_station[("http://b", "key-B")] == 1
        assert tracker.peak_total == 2, "跨站必须并行，否则本策略把池退化成单路"
        # 4 发 × 0.15s：纯串行≈0.6s，两站并行≈0.3s
        assert spent < 0.55, f"耗时 {spent:.2f}s 说明并未跨站并行"
    finally:
        pool.close()


# ---------------- 3) 配置可退回旧行为 ----------------

def test_max_per_station_relaxes_the_gate(monkeypatch):
    """max_per_station=2 时同站允许 2 路在飞：证明约束由配置把守，不是写死。"""
    tracker = _Tracker(hold=0.15)
    monkeypatch.setattr(orch, "LLMClient", _fake_client(tracker))
    pool = _pool([("A", "http://a", "m1,m2")], max_per_station=2)
    try:
        pool.collect(["A:m1", "A:m2"], "p")
        assert tracker.peak_station[("http://a", "key-A")] == 2
    finally:
        pool.close()


# ---------------- 4) 整体限时按站数放宽 ----------------

def test_effective_timeout_accounts_for_station_serialization():
    """6 名 / 2 站 / 每站 1 ⇒ 实际并行 2 路 = 3 批 ⇒ 3×120+30；旧口径只有 2 批会误杀尾部。"""
    pool = _pool([("A", "http://a", "m1,m2,m3"), ("B", "http://b", "m1,m2,m3")])
    try:
        names = ["A:m1", "A:m2", "A:m3", "B:m1", "B:m2", "B:m3"]
        assert pool._batch_concurrency(len(names), names) == 2, "min(max_workers=3, 1×2站)"
        assert pool._effective_timeout(len(names), None, names) == 3 * 120 + 30
        # 不传 workers ⇒ 退回旧口径（既有断言/调用方行为逐字不变）
        assert pool._effective_timeout(len(names), None) == 2 * 120 + 30
        # 显式传入永远优先
        assert pool._effective_timeout(len(names), 0.3, names) == 0.3
    finally:
        pool.close()


def test_single_station_batch_widens_timeout_the_most():
    """整批都在同站 ⇒ 并行度 1，限时必须按人数全额放宽，否则排队必被误判超时。"""
    pool = _pool([("A", "http://a", "m1,m2,m3")])
    try:
        names = ["A:m1", "A:m2", "A:m3"]
        assert pool._batch_concurrency(len(names), names) == 1
        assert pool._effective_timeout(len(names), None, names) == 3 * 120 + 30
    finally:
        pool.close()


# ---------------- 5) 回退不得在门内递归 ----------------

def test_same_station_fallback_does_not_self_deadlock(monkeypatch):
    """同站回退必须跑得完：首个模型瞬时失败 ⇒ 换同站第二个模型成功续跑。

    这条断言的"牙齿"在于**门只包一次 chat**：若把门挪到包住整条回退链（持着 A 站的门
    再打 A 站的另一个模型），每站配额 1 时第二次 acquire 就是自锁，本用例会挂在 5s 看门狗上。
    """
    tracker = _Tracker(hold=0.02)
    monkeypatch.setattr(orch, "LLMClient", _fake_client(tracker, fail_once=[("http://a", "key-A")]))
    pool = _pool([("A", "http://a", "m1,m2")])
    box: dict = {}

    def _run():
        box["out"] = pool.ask_result("A:m1", "p")

    try:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(5.0)
        assert not t.is_alive(), "同站回退挂死：回退请求被原站的门挡在自己外面"
        out = box["out"]
        assert out["ok"] is True, f"同站回退应成功续跑：{out}"
        assert tracker.calls[("http://a", "key-A")] == 2, "首发失败 + 同站回退各一次"
    finally:
        pool.close()


def test_reverse_cross_station_fallback_does_not_deadlock(monkeypatch):
    """两个单模型站互为唯一回退目标、且各自首发改失败 ⇒ 反向回退不得死锁。

    持门递归的写法在这里会 A 持门等 B、B 持门等 A，永久挂住；本用例用硬超时把它
    变成一次失败（线程没在 5s 内回来即判挂死）。
    """
    tracker = _Tracker(hold=0.02)
    monkeypatch.setattr(
        orch, "LLMClient",
        _fake_client(tracker, fail_once=[("http://a", "key-A"), ("http://b", "key-B")]),
    )
    pool = _pool([("A", "http://a", "m1"), ("B", "http://b", "m1")])
    result: dict = {}

    def _run():
        result["out"] = pool.ask_many(["A", "B"], "p")

    try:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(5.0)
        assert not t.is_alive(), "反向跨站回退挂死：回退请求在门内等待"
        assert "超时" not in result["out"], result["out"]
    finally:
        pool.close()


# ---------------- 6) 延迟探测让路 ----------------

def test_probe_yields_when_station_is_busy(monkeypatch):
    """站被真实派工占着时，探测最多等 _PROBE_GATE_WAIT 就放弃（绝不陪等 120s）。"""
    tracker = _Tracker(hold=0.0)
    monkeypatch.setattr(orch, "LLMClient", _fake_client(tracker))
    monkeypatch.setattr(latency_mod, "_PROBE_GATE_WAIT", 0.2)
    pool = _pool([("A", "http://a", "m1")])
    try:
        station = ("http://a", "key-A")
        gate = pool._station_gate(station)
        assert gate.acquire(timeout=1), "夹具：先占住站门模拟在飞请求"
        t0 = time.monotonic()
        assert pool._probe_one("A") is None, "门被占 ⇒ 本轮跳过该工人"
        assert time.monotonic() - t0 < 1.5, "探测被在飞请求挂住，失去让路语义"
        assert tracker.calls.get(station, 0) == 0, "让路时不该发出探测请求"
        gate.release()
        assert pool._probe_one("A") is not None, "门空出来就该正常测速"
    finally:
        pool.close()


# ---------------- 7) 真实并发口径回给外部主 ----------------

def test_health_reports_effective_concurrency():
    """`health` 必须带出 effective：主靠它决定"一批点几个、要不要跨站"。

    2 站 × 每站 1 且 max_workers=3 ⇒ 真实并行度 2（不是 3）。少了这个字段，主会按
    max_workers 点名 3 个同站模型，第三个在站门口排队、还被误当成"这个模型慢"。
    """
    from agent.bridge import Bridge

    cfg = Config(_interpolate({
        "agents": [
            {"name": "A", "base_url": "http://a", "api_key": "ka", "models": "m1,m2"},
            {"name": "B", "base_url": "http://b", "api_key": "kb", "models": "m1,m2"},
        ],
        "llm": {"base_url": "u", "api_key": "k", "model": "m"},
        "collaboration": {"max_workers": 3, "max_per_station": 1, "latency_probe": False},
    }))
    b = Bridge(cfg)
    try:
        out = b.handle({"cmd": "health"})
        assert out["concurrency"] == {
            "stations": 2, "max_workers": 3, "max_per_station": 1, "effective": 2,
        }, out.get("concurrency")
    finally:
        b.workers.close()


# ---------------- 8) 配置容错（与其余数值项同口径） ----------------
@pytest.mark.parametrize("bad", ["many", 0, -1, None])
def test_bad_max_per_station_falls_back_to_one(bad):
    """max_per_station 写错/写 0 不能崩启动，也不能变成"无门"。"""
    cfg = Config(_interpolate({
        "agents": [{"name": "A", "base_url": "http://a", "api_key": "k", "models": "m1"}],
        "collaboration": {"max_per_station": bad},
    }))
    assert cfg.max_per_station == 1
