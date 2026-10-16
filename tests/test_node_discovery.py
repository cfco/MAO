"""免费节点自动发现（node_discovery）离线单测：全程 mock 网络，不触外网。

覆盖：叠加层增量计算（merge_delta / _reconcile）、kilo 适配器解析、
发现器 run_once 写叠加层且幂等、apply_overlay 在 load_config 路径上的叠加、
以及 WorkerPool.sync_free_nodes 热同步进运行中的池子。
"""

import os
from pathlib import Path

import pytest

from agent.config import Config, _interpolate
from agent.core.node_discovery import (
    FreeModel,
    FreeNodeDiscoverer,
    KiloAdapter,
    _reconcile,
    apply_overlay,
    load_overlay,
    merge_delta,
    save_overlay,
)
from agent.core.orchestrator import WorkerPool

# ---------- 基础增量 ----------


def test_merge_delta_removes_retired_keeps_hash_and_appends_added():
    base = "m1@256k,#m2@256k,m3@256k"
    out = merge_delta(base, {"added": ["m4@128k"], "retired": ["m1"]})
    assert out == "#m2@256k,m3@256k,m4@128k"
    # 手动 # 屏蔽的 m2 仍在；retired 的 m1 被移除；added 的 m4 追加


def test_merge_delta_does_not_reenable_manual_hash():
    # 已人工 # 屏蔽的模型不应被 added 自动复活
    base = "#m1@256k,m2@256k"
    out = merge_delta(base, {"added": ["m1@256k"], "retired": []})
    assert out == "#m1@256k,m2@256k"


def test_merge_delta_skips_duplicate_added():
    base = "m1@256k"
    out = merge_delta(base, {"added": ["m1@256k", "m2@256k"], "retired": []})
    assert out == "m1@256k,m2@256k"


def test_reconcile_relative_to_registry():
    # registry 原始行只有 m1；在线有 m1,m2 → added m2；m1 在线存在 → 不 retired
    delta = _reconcile("m1@256k,#m0@256k", [FreeModel("m1"), FreeModel("m2", 262144)])
    assert delta["added"] == ["m2@262144"]
    assert delta["retired"] == []  # m0 是人工 # 屏蔽，不在活跃集，不应被 retired


def test_reconcile_retires_vanished_active():
    delta = _reconcile("m1@256k,m2@256k", [FreeModel("m1")])
    assert delta["retired"] == ["m2"]


# ---------- 适配器 ----------


def test_kilo_adapter_filters_free_suffix(monkeypatch):
    sample = {
        "data": [
            {"id": "foo:free", "context_length": 262144},
            {"id": "bar", "context_length": 1000},  # 非免费档
            {"id": "baz:free", "context_length": 8000},
        ]
    }
    import agent.core.node_discovery as nd

    monkeypatch.setattr(nd, "_http_get_json", lambda url, key, timeout: sample)
    out = KiloAdapter().fetch("", "", 5)
    assert [m.model for m in out] == ["foo:free", "baz:free"]
    assert out[0].ctx == 262144


# ---------- 发现器 run_once + 叠加层读写 ----------


def _cfg_with_registry(tmp_path, agents, **collab):
    # 健康/延迟档案指向 tmp，避免污染真实 data/；同时避免 WorkerPool 建池起后台线程打网络
    os.environ["MAO_HEALTH_FILE"] = str(tmp_path / "h.json")
    os.environ["MAO_LATENCY_FILE"] = str(tmp_path / "l.json")
    registry = tmp_path / "model_registry.txt"
    registry.write_text("NODE_A_MODELS=m1@256k\nNODE_B_MODELS=k1:free@250k\n", encoding="utf-8")
    overlay = tmp_path / "free_nodes.json"
    data = {
        "agents": agents,
        "llm": {"timeout": 5, "max_retries": 0},
        "collaboration": {
            "free_node_discovery": True,
            "free_node_interval": 0,  # 关定时，run_once 单独调用
            "free_node_timeout": 2,
            **collab,
        },
    }
    cfg = Config(_interpolate(data), raw=data)
    # 直接构造时 models_env 不会从 ${VAR} 推导，这里手动标上去模拟
    for p in cfg.agent_profiles:
        p.models_env = "NODE_A_MODELS" if p.name == "node-a" else "NODE_B_MODELS"
        p.discovery = "kilo"
    return cfg, registry, overlay


def test_discoverer_run_once_writes_overlay_and_idempotent(tmp_path, monkeypatch):
    cfg, registry, overlay = _cfg_with_registry(
        tmp_path, [{"name": "node-a", "base_url": "x", "api_key": "k", "models": "m1@256k"}]
    )

    class FakeKilo:
        name = "kilo"

        def fetch(self, base_url, api_key, timeout):
            return [FreeModel("m1"), FreeModel("m2", 262144)]

    disc = FreeNodeDiscoverer(cfg, overlay, adapters={"kilo": FakeKilo()})
    report = disc.run_once()
    assert report["stations"]["NODE_A_MODELS"]["added"] == ["m2@262144"]
    assert report["updated"] is True
    written = load_overlay(overlay)
    assert written["NODE_A_MODELS"]["added"] == ["m2@262144"]

    # 第二次：registry 没变 → 不应再写（幂等）
    report2 = disc.run_once()
    assert report2["updated"] is False


def test_discoverer_run_once_fetch_failure_skips_station(tmp_path, monkeypatch):
    import agent.core.node_discovery as nd

    cfg, registry, overlay = _cfg_with_registry(
        tmp_path, [{"name": "node-a", "base_url": "x", "api_key": "k", "models": "m1@256k"}]
    )
    monkeypatch.setattr(nd, "_http_get_json", lambda url, key, timeout: None)
    disc = FreeNodeDiscoverer(cfg, overlay, adapters={"kilo": KiloAdapter()})
    report = disc.run_once()
    assert report["stations"]["NODE_A_MODELS"]["status"] == "skip"
    assert not overlay.exists()  # 没写任何叠加层


# ---------- apply_overlay（load_config 路径） ----------


def test_apply_overlay_patches_env(tmp_path):
    os.environ["NODE_A_MODELS"] = "m1@256k,#m2@256k"
    overlay = tmp_path / "free_nodes.json"
    save_overlay(overlay, {"NODE_A_MODELS": {"added": ["m3@128k"], "retired": ["m1"]}})
    apply_overlay(overlay)
    assert os.environ["NODE_A_MODELS"] == "#m2@256k,m3@128k"
    del os.environ["NODE_A_MODELS"]


def test_apply_overlay_no_file_is_noop():
    os.environ["NODE_A_MODELS"] = "m1@256k"
    # 叠加层文件缺失 → 不改动 env（也不抛异常）
    apply_overlay(Path("__nonexistent_free_nodes__.json"))
    assert os.environ["NODE_A_MODELS"] == "m1@256k"
    del os.environ["NODE_A_MODELS"]


# ---------- WorkerPool 热同步 ----------


@pytest.fixture
def disabled_probes(monkeypatch):
    monkeypatch.setenv("MAO_LATENCY_PROBE", "0")
    monkeypatch.setenv("MAO_FREE_NODE_PROBE", "0")


def test_sync_free_nodes_adds_and_removes(tmp_path, disabled_probes):
    cfg, registry, overlay = _cfg_with_registry(
        tmp_path, [{"name": "node-a", "base_url": "x", "api_key": "k", "models": "m1@256k"}]
    )
    pool = WorkerPool(cfg)
    before = set(pool.profiles)
    assert before == {"node-a"}  # m1（单模型保留站名）

    applied = pool.sync_free_nodes(
        {"NODE_A_MODELS": {"added": ["m2@128k", "m3@256k"], "retired": ["m1"]}}
    )
    after = set(pool.profiles)
    assert after == {"node-a:m2", "node-a:m3"}
    assert applied["added"] == ["node-a:m2", "node-a:m3"]
    assert applied["removed"] == ["node-a"]

    # 再次同步：已存在的模型不重复加
    applied2 = pool.sync_free_nodes({"NODE_A_MODELS": {"added": ["m2@128k"], "retired": []}})
    assert applied2["added"] == []
    pool.close()


def test_sync_free_nodes_does_not_revive_manually_blocked(tmp_path, disabled_probes):
    """铁律 1 的端到端回归：人工 `#` 屏蔽、在线仍存在的模型，不能被热同步复活。

    替换掉的旧用例把"被屏蔽的模型"直接塞成池里已有的 profile —— 真实场景里
    `_split_models` 根本不会为 `#` 项建 profile，所以那条断言恒真、护不住铁律。
    缺陷本体：`_bare("#m2@256k")` 保留 `#`，而在线清单是裸名，`_reconcile` 直接比
    就把屏蔽模型算进 added；落盘侧 merge_delta 有 lstrip 兜住，热同步侧（只看当前池
    的 profile）兜不住 → 屏蔽被悄悄复活。
    """
    cfg, _, overlay = _cfg_with_registry(
        tmp_path,
        [{"name": "node-a", "base_url": "x", "api_key": "k", "models": "m1@256k,#m2@256k"}],
    )
    cfg.model_registry_path.write_text("NODE_A_MODELS=m1@256k,#m2@256k\n", encoding="utf-8")

    class FakeKilo:
        name = "kilo"

        def fetch(self, base_url, api_key, timeout):
            return [FreeModel("m1"), FreeModel("m2", 262144)]

    disc = FreeNodeDiscoverer(cfg, overlay, adapters={"kilo": FakeKilo()})
    report = disc.run_once()
    added = report["stations"]["NODE_A_MODELS"]["added"]
    assert "m2@262144" not in added, f"人工屏蔽的 m2 不该被算成新增：{added}"

    pool = WorkerPool(cfg)
    try:
        applied = pool.sync_free_nodes(load_overlay(overlay))
        assert "node-a:m2" not in pool.profiles, "被屏蔽模型不得进入运行池"
        assert applied["added"] == []
    finally:
        pool.close()


def test_reconcile_ignores_manually_blocked_names():
    """`_reconcile` 两侧比较都必须剥掉 `#` 前缀，否则屏蔽项永远"看起来不存在"。"""
    delta = _reconcile("m1@256k,#m2@256k", [FreeModel("m1"), FreeModel("m2")])
    assert delta["added"] == []
    assert delta["retired"] == []  # m2 是人工屏蔽，不在活跃集，不该被算成下线


class _OneModelAdapter:
    name = "kilo"

    def __init__(self):
        self.calls = 0

    def fetch(self, base_url, api_key, timeout):
        self.calls += 1
        return [FreeModel("m1")]


def test_manual_refresh_still_works_when_interval_is_zero(tmp_path, monkeypatch):
    """interval<=0 只该关掉"自动排轮"，手动 refresh_nodes 必须照常可用。

    旧写法把 `_interval > 0` 混进 enabled()，而 refresh_now 也先查 enabled() ——
    配成 0 等于连手动刷新一起废掉，与 config.yaml 的承诺正好相反。
    """
    monkeypatch.delenv("MAO_FREE_NODE_PROBE", raising=False)  # 本轮要真跑发现，只关 kill switch
    cfg, _, overlay = _cfg_with_registry(
        tmp_path, [{"name": "node-a", "base_url": "x", "api_key": "k", "models": "m1@256k"}]
    )  # helper 里 free_node_interval=0
    adapter = _OneModelAdapter()
    disc = FreeNodeDiscoverer(cfg, overlay, adapters={"kilo": adapter})
    assert disc.enabled() is True and disc.due() is False, "interval=0 应只关自动排轮"
    report = disc.refresh_now()
    assert adapter.calls == 1 and report["stations"]["NODE_A_MODELS"]["status"] == "ok"
    assert disc.status()["scheduled"] is False


def test_maybe_start_rolls_back_the_schedule_when_thread_fails(tmp_path, monkeypatch):
    """线程起不来时周期起算点也要回滚，否则这一轮发现被丢掉还要再等一整个 interval。"""
    monkeypatch.delenv("MAO_FREE_NODE_PROBE", raising=False)
    cfg, _, overlay = _cfg_with_registry(
        tmp_path, [{"name": "node-a", "base_url": "x", "api_key": "k", "models": "m1@256k"}]
    )
    disc = FreeNodeDiscoverer(cfg, overlay, adapters={"kilo": _OneModelAdapter()})
    disc._interval = 60.0
    monkeypatch.setattr(disc, "due", lambda: True)

    def boom(*a, **kw):
        raise RuntimeError("can't start thread")

    monkeypatch.setattr("agent.core.node_discovery.threading.Thread", boom)
    assert disc.maybe_start() is False
    assert disc._running is False
    assert disc._last_start == disc._last_finish, "占位失败后必须退回上一轮结束时间，立即可重试"


def test_load_config_applies_overlay_from_the_configured_path(tmp_path, monkeypatch):
    """`collaboration.free_nodes_file` 必须同时被写侧与读侧认（曾只认 env，写 A 读 B）。"""
    from agent.config import load_config

    monkeypatch.delenv("MAO_FREE_NODES_FILE", raising=False)
    overlay = tmp_path / "my_overlay.json"
    save_overlay(overlay, {"NODE_A_MODELS": {"added": ["newm@128k"], "retired": ["gone@256k"]}})
    monkeypatch.setenv("NODE_A_MODELS", "m1@256k,gone@256k")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "agents:\n"
        "  - name: node-a\n"
        "    base_url: http://x\n"
        "    api_key: k\n"
        "    models: ${NODE_A_MODELS}\n"
        "collaboration:\n"
        f"  free_nodes_file: {overlay.as_posix()}\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    models = {p.model for p in cfg.agent_profiles}
    assert cfg.free_nodes_path == overlay
    assert "newm" in models, f"叠加层没被应用（读侧忽略了配置的自定义路径）：{sorted(models)}"
    assert not any(m.startswith("gone") for m in models), sorted(models)
