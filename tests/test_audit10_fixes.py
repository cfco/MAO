from __future__ import annotations

import asyncio
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.config import Config
from agent.core.constants import ST_TIMEOUT
from agent.core.health import ModelHealth
from agent.core.llm import LLMClient
from agent.core.node_discovery import FreeModel, FreeNodeDiscoverer, load_overlay
from agent.core.orchestrator import WorkerPool
from agent.core.parallelism import _CANCEL_EVENT, DispatchCancelled, is_dispatch_cancelled
from agent.model_registry import retire_model
from agent.tools.mcp_client import McpConnection
from agent.tools.shell_safety import _check_command_safety


def _pool(tmp_path: Path, monkeypatch, data: dict) -> WorkerPool:
    monkeypatch.setenv("MAO_HEALTH_FILE", str(tmp_path / "health.json"))
    monkeypatch.setenv("MAO_LATENCY_FILE", str(tmp_path / "latency.json"))
    monkeypatch.setenv("MAO_MODEL_REGISTRY_FILE", str(tmp_path / "registry.txt"))
    monkeypatch.setenv("MAO_FREE_NODES_FILE", str(tmp_path / "free_nodes.json"))
    monkeypatch.setenv("MAO_LATENCY_PROBE", "0")
    monkeypatch.setenv("MAO_FREE_NODE_PROBE", "0")
    cfg = Config(data)
    return WorkerPool(cfg)


def _station_pool(tmp_path: Path, monkeypatch) -> WorkerPool:
    return _pool(
        tmp_path,
        monkeypatch,
        {
            "agents": [
                {
                    "name": "a",
                    "base_url": "https://a.example/v1",
                    "api_key": "ka",
                    "models": "m1,m2,m3",
                },
                {"name": "b", "base_url": "https://b.example/v1", "api_key": "kb", "model": "m1"},
            ],
            "llm": {"timeout": 5, "max_retries": 0},
            "collaboration": {
                "max_workers": 2,
                "max_per_station": 1,
                "fallback_models": 0,
                "free_node_discovery": False,
            },
        },
    )


def test_zero_max_retries_survives_pool_construction(tmp_path, monkeypatch):
    pool = _station_pool(tmp_path, monkeypatch)
    try:
        assert pool._max_retries == 0
    finally:
        pool.close()


def test_station_skew_uses_queue_waves_and_interleaves_stations(tmp_path, monkeypatch):
    pool = _station_pool(tmp_path, monkeypatch)
    try:
        names = ["a:m1", "a:m2", "a:m3", "b"]
        assert pool._batch_waves(4, names) == 3
        assert pool._station_ordered(names) == ["a:m1", "b", "a:m2", "a:m3"]
        expected = 3 * 5 + 30
        assert pool._effective_timeout(4, None, names) == expected
    finally:
        pool.close()


def test_overall_timeout_sets_cancellation_for_running_worker(tmp_path, monkeypatch):
    pool = _station_pool(tmp_path, monkeypatch)
    started = threading.Event()
    release = threading.Event()
    observed = threading.Event()
    seen = []

    def fake_ask(worker, prompt, system=None, memo=None):
        started.set()
        release.wait(5)
        seen.append(is_dispatch_cancelled())
        observed.set()
        return {"worker": worker, "ok": False, "status": ST_TIMEOUT, "answer": "", "error": "late"}

    monkeypatch.setattr(pool, "ask_result", fake_ask)
    try:
        out = pool._run_parallel(["b"], "p", None, 0.05)
        assert started.wait(2)
        assert out["b"] is None
        release.set()
        assert observed.wait(2)
        assert seen == [True]
    finally:
        release.set()
        pool.close()


def test_cancelled_station_gate_does_not_call_client(tmp_path, monkeypatch):
    pool = _station_pool(tmp_path, monkeypatch)
    client = MagicMock()
    event = threading.Event()
    event.set()
    token = _CANCEL_EVENT.set(event)
    try:
        with pytest.raises(DispatchCancelled):
            pool._chat_gated(client, ("https://a.example/v1", "ka"), [])
        client.chat_once.assert_not_called()
    finally:
        _CANCEL_EVENT.reset(token)
        pool.close()


def test_client_lease_defers_close_until_release():
    client = LLMClient("http://x", "k", "m")
    fake = MagicMock()
    client.client = fake
    assert client.acquire_call() is True
    client.close_when_idle()
    fake.close.assert_not_called()
    client.release_call()
    fake.close.assert_called_once()


def test_health_query_refreshes_after_another_instance_writes(tmp_path):
    path = tmp_path / "health.json"
    writer = ModelHealth(path, None, 7)
    reader = ModelHealth(path, None, 7)
    assert reader.quarantined("worker") is False
    writer.record_failure("worker", "m", "NODE_MODELS")
    assert reader.quarantined("worker") is True


def test_registry_concurrent_retirements_keep_both_marks(tmp_path):
    path = tmp_path / "registry.txt"
    path.write_text("NODE_MODELS=m1@8k,m2@8k\n", encoding="utf-8")
    threads = [
        threading.Thread(target=retire_model, args=(path, "NODE_MODELS", m)) for m in ("m1", "m2")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    text = path.read_text(encoding="utf-8")
    assert "#m1@8k" in text and "#m2@8k" in text


def test_ledger_cas_exhaustion_does_not_write_stale_snapshot(tmp_path, monkeypatch):
    from agent.core import ledger as ledger_mod

    class StaleLedger(ledger_mod.AtomicJsonLedger):
        def __init__(self, path):
            super().__init__(path)
            self.reads = 0

        def _ensure_loaded(self):
            return None

        def _snapshot_locked(self):
            return "stale"

        def _disk_for_merge(self):
            self.reads += 1
            return {}, f"disk-{self.reads}"

        def _read_disk_text(self):
            return "newer"

    write = MagicMock()
    monkeypatch.setattr(ledger_mod, "atomic_write_text", write)
    StaleLedger(tmp_path / "health.json").write()
    write.assert_not_called()


def _discoverer(tmp_path: Path, env_var: str, adapter) -> FreeNodeDiscoverer:
    registry = tmp_path / "registry.txt"
    registry.write_text(f"{env_var}=base:free@8k\n", encoding="utf-8")
    cfg = SimpleNamespace(
        free_node_interval=0,
        free_node_timeout=1,
        free_node_discovery=True,
        model_registry_path=registry,
        agent_profiles=[
            SimpleNamespace(
                models_env=env_var,
                base_url="https://api.kilo.ai/v1",
                api_key="",
                discovery="fake",
            )
        ],
    )
    return FreeNodeDiscoverer(cfg, tmp_path / "overlay.json", adapters={"fake": adapter})


def test_discovery_can_retire_and_restore_models(tmp_path):
    class Sequence:
        name = "fake"
        host = ""
        free_marker = ":free"

        def __init__(self):
            self.calls = 0

        def fetch(self, *args):
            self.calls += 1
            if self.calls == 1:
                return [FreeModel("base:free", 8192), FreeModel("new:free", 8192)]
            if self.calls == 2:
                return [FreeModel("base:free", 8192)]
            return [FreeModel("base:free", 8192), FreeModel("new:free", 8192)]

    disc = _discoverer(tmp_path, "NODE_X_MODELS", Sequence())
    disc.run_once()
    assert "new:free@8192" in load_overlay(disc.overlay_path)["NODE_X_MODELS"]["added"]
    disc.run_once()
    delta = load_overlay(disc.overlay_path)["NODE_X_MODELS"]
    assert delta == {"added": [], "retired": ["new:free"]}
    disc.run_once()
    delta = load_overlay(disc.overlay_path)["NODE_X_MODELS"]
    assert delta == {"added": ["new:free@8192"], "retired": []}


def test_discovery_concurrent_stations_preserve_both(tmp_path):
    class OneStation:
        name = "fake"
        host = ""
        free_marker = ":free"

        def __init__(self, model):
            self.model = model

        def fetch(self, *args):
            return [FreeModel(self.model, 8192), FreeModel(f"{self.model}-new:free", 8192)]

    overlay = tmp_path / "overlay.json"
    first = _discoverer(tmp_path, "NODE_X_MODELS", OneStation("x:free"))
    second_cfg = SimpleNamespace(
        free_node_interval=0,
        free_node_timeout=1,
        free_node_discovery=True,
        model_registry_path=tmp_path / "registry.txt",
        agent_profiles=[
            SimpleNamespace(
                models_env="NODE_Y_MODELS",
                base_url="https://api.kilo.ai/v1",
                api_key="",
                discovery="fake",
            )
        ],
    )
    (tmp_path / "registry.txt").write_text(
        "NODE_X_MODELS=base:free@8k\nNODE_Y_MODELS=base:free@8k\n", encoding="utf-8"
    )
    second = FreeNodeDiscoverer(second_cfg, overlay, adapters={"fake": OneStation("y:free")})
    threads = [threading.Thread(target=first.run_once), threading.Thread(target=second.run_once)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    data = load_overlay(overlay)
    assert set(data) == {"NODE_X_MODELS", "NODE_Y_MODELS"}


def test_sync_free_nodes_recovers_after_station_becomes_empty(tmp_path, monkeypatch):
    pool = _pool(
        tmp_path,
        monkeypatch,
        {
            "agents": [
                {
                    "name": "node-x",
                    "base_url": "https://api.kilo.ai/v1",
                    "api_key": "k",
                    "models": "m1:free@8k",
                    "tags": "code",
                }
            ],
            "collaboration": {"free_node_discovery": False},
        },
    )
    pool.cfg.agent_profiles[0].models_env = "NODE_X_MODELS"
    pool._env_of = {p.name: p.models_env for p in pool.cfg.agent_profiles}
    try:
        pool.sync_free_nodes({"NODE_X_MODELS": {"added": [], "retired": ["m1:free"]}})
        assert pool.profiles == {}
        applied = pool.sync_free_nodes({"NODE_X_MODELS": {"added": ["m2:free@8k"], "retired": []}})
        assert applied["added"] == ["node-x:m2:free"]
        assert pool.profiles["node-x:m2:free"].tags == "code"
    finally:
        pool.close()


def test_bad_config_warning_does_not_echo_value(capsys):
    secret = "sk-REALSECRETVALUE123456"
    assert Config.as_int(secret, "LLM_TIMEOUT", 120) == 120
    assert secret not in capsys.readouterr().err


def test_bool_config_rejects_non_binary_numbers():
    assert Config.as_bool(float("nan"), "nan", False) is False
    assert Config.as_bool(float("inf"), "inf", False) is False
    assert Config.as_bool(2, "two", False) is False
    assert Config.as_bool(-1, "minus", False) is False
    assert Config.as_bool(1, "one", False) is True
    assert Config.as_bool(0, "zero", True) is False


def test_openai_client_handles_bracketed_ipv6_no_proxy(monkeypatch):
    value = "localhost,127.0.0.1,[::1]"
    monkeypatch.setenv("NO_PROXY", value)
    if os.name != "nt":
        monkeypatch.setenv("no_proxy", value)
    client = LLMClient("http://x", "k", "m")
    client.close()
    assert os.environ["NO_PROXY"] == value
    if os.name != "nt":
        assert os.environ["no_proxy"] == value


def test_transparent_command_wrappers_are_scanned():
    assert _check_command_safety("cmd /c call del /f /q x") is not None
    assert _check_command_safety("command rm -f x") is not None


def test_redirect_check_is_quote_aware():
    assert _check_command_safety('echo "text > /tmp/x"') is None
    assert _check_command_safety("python -c \"print('x > /tmp/y')\"") is None
    assert _check_command_safety('echo x > "/tmp/outside"') is not None


def test_skill_timeout_kills_process_tree(tmp_path, monkeypatch):
    from agent import skills_manager as sm
    from agent.tools import builtin

    skill = tmp_path / "s1"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: s1\ndescription: d\n---\n", encoding="utf-8")
    (skill / "scripts" / "slow.py").write_text("pass\n", encoding="utf-8")
    mgr = sm.SkillManager(tmp_path)
    mgr.scan()

    class Proc:
        stdout = None
        stderr = None
        returncode = None

        def __init__(self):
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("slow.py", timeout)
            return b"", b""

    proc = Proc()
    kill = MagicMock()
    monkeypatch.setattr(sm.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(builtin, "_kill_process_tree", kill)
    result = mgr.run_script("s1", "slow.py")
    assert getattr(result, "ok", True) is False
    kill.assert_called_once_with(proc)


def test_mcp_shutdown_cleans_transport_when_session_exit_raises():
    class SessionCM:
        async def __aexit__(self, *args):
            raise RuntimeError("session close failed")

    class TransportCM:
        def __init__(self):
            self.exited = False

        async def __aexit__(self, *args):
            self.exited = True

    conn = McpConnection("t", {"name": "t", "transport": "stdio", "command": "x"})
    transport = TransportCM()
    conn._session_cm = SessionCM()
    conn._transport_cm = transport
    asyncio.run(conn._shutdown_once())
    assert transport.exited is True
