"""模型健康 + 配置分层的回归测试：全部离线，无需 API key / 网络。

覆盖两块（2026-09-21 需求）：
1) 配置分层：.env.example 承载接口与模型清单（基础层，运行时真实加载），
   .env 只放 key（覆盖层）；优先级 shell > .env > .env.example；
   .env.example 变化要让配置缓存失效。
2) 模型健康：一次终态失败 → 当日隔离不再派工（缓存命中不受影响）；
   连续 retire_days 个"运行日"（程序实际启动过的天，周末没开机不计入也不打断）
   都失败 → 自动在 .env.example 对应行给模型标 # 下线
   （LLM_MODEL 兜底行不自动动，人工决定）。
"""
from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path

import pytest

import agent.config as ac
import agent.core.orchestrator as orch
from agent.config import Config, _interpolate, load_config
from agent.core.health import ModelHealth, _today
from agent.core.llm import LLMError
from agent.core.orchestrator import WorkerPool

# ---------------- 公共夹具 ----------------


@pytest.fixture
def cfg_home(tmp_path, monkeypatch):
    """把配置根目录挪到 tmp_path，并重置进程内缓存/环境变量注入记录。

    测试结束后 os.environ 整体还原：dotenv 注入走的是直接赋值，
    monkeypatch 管不到，不清理会把 NODE_* 泄漏给后续测试。
    """
    snapshot = dict(os.environ)
    # 先摘掉此前由 dotenv 注入的变量（如 test_config_contract 加载真实仓库配置时
    # 注入的 NODE_*/LLM_*）：不清的话会被 _load_dotenv 当作"shell 预置"最高优先级，
    # tmp 里的分层文件永远覆盖不掉，池子插值到的是真仓库的模型清单。
    for leaked in ac._dotenv_keys:
        os.environ.pop(leaked, None)
    monkeypatch.setattr(ac, "ROOT", tmp_path)
    monkeypatch.setattr(ac, "_cfg_cache", None)
    monkeypatch.setattr(ac, "_cfg_cache_mtime", 0.0)
    monkeypatch.setattr(ac, "_cfg_cache_dotenv_mtime", 0.0)
    monkeypatch.setattr(ac, "_cfg_cache_example_mtime", 0.0)
    monkeypatch.setattr(ac, "_dotenv_keys", set())
    monkeypatch.setattr(ac, "_missing_logged", set())
    yield tmp_path
    os.environ.clear()
    os.environ.update(snapshot)


def _day(offset: int) -> str:
    return (date.fromisoformat(_today()) + timedelta(days=offset)).isoformat()


CONFIG_YAML = """
agents:
  - name: node-a
    base_url: ${NODE_A_ENDPOINT}
    api_key: ${NODE_A_KEY}
    models: ${NODE_A_MODELS}
llm:
  base_url: ${LLM_ENDPOINT:-http://fallback}
  api_key: ${LLM_API_KEY}
  model: ${LLM_MODEL:-fb-model}
"""


def _write_layers(home, models_a="m-one@128k,m-two@256k", env_lines=("NODE_A_KEY=sk-real",)):
    (home / "config.yaml").write_text(CONFIG_YAML, encoding="utf-8")
    (home / ".env.example").write_text(
        "# 分层基础：接口与模型清单\n"
        "NODE_A_ENDPOINT=https://a.example/v1\n"
        "NODE_A_KEY=\n"
        f"NODE_A_MODELS={models_a}\n"
        "LLM_ENDPOINT=https://a.example/v1\n"
        "LLM_API_KEY=\n"
        "LLM_MODEL=m-one\n",
        encoding="utf-8",
    )
    (home / ".env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")


# ---------------- 1) 配置分层 ----------------


def test_layers_example_base_env_override(cfg_home):
    _write_layers(cfg_home)
    cfg = load_config()
    assert cfg.agent_profiles, "接口/模型在 .env.example、key 在 .env，池应能建起来"
    p = cfg.profile("node-a:m-one")
    assert p is not None and p.base_url == "https://a.example/v1"
    assert p.api_key == "sk-real", ".env 的 key 必须覆盖 .env.example 的空占位"
    assert p.models_env == "NODE_A_MODELS", "要能反查模型清单所在变量（健康下线改写用）"
    assert cfg.llm.get("api_key") == "", "LLM_API_KEY 只有 .env.example 空占位 → 插值为空串"


def test_shell_env_wins_over_both_dotenv_layers(cfg_home, monkeypatch):
    monkeypatch.setenv("NODE_A_ENDPOINT", "http://shell.wins/v1")
    _write_layers(cfg_home)
    # shell 预置发生在 dotenv 加载之前也要成立：dotenv 只更新"自己注入过"的键
    os.environ["NODE_A_KEY"] = "sk-shell"
    cfg = load_config()
    p = cfg.profile("node-a:m-one")
    assert p.base_url == "http://shell.wins/v1"
    assert p.api_key == "sk-shell", "shell 已导出的 key 不被 .env 覆盖"


def test_missing_env_file_still_loads_from_example(cfg_home):
    _write_layers(cfg_home)
    (cfg_home / ".env").unlink()
    cfg = load_config()
    p = cfg.profile("node-a:m-one")
    assert p is not None and p.api_key == "", "没有 .env 时池照常构建，key 为空占位"


def test_env_example_change_invalidates_cache(cfg_home):
    _write_layers(cfg_home)
    cfg1 = load_config()
    assert load_config() is cfg1, "三份文件都没动时应命中缓存"
    # 只改 .env.example（比如下线了 m-two），并显式推后 mtime 保证跨平台确定性
    ex = cfg_home / ".env.example"
    ex.write_text(
        ex.read_text(encoding="utf-8").replace("NODE_A_MODELS=m-one@128k,m-two@256k", "NODE_A_MODELS=m-one@128k"),
        encoding="utf-8",
    )
    st = ex.stat()
    os.utime(ex, (st.st_atime + 5, st.st_mtime + 5))
    cfg2 = load_config()
    assert cfg2.profile("node-a:m-two") is None, ".env.example 变化必须让缓存失效重载"
    # 删剩单模型后按规则改名为站名（不再带 :模型 后缀）
    assert cfg2.profile("node-a") is not None


def test_hash_blocked_model_untouched_by_layering(cfg_home):
    _write_layers(cfg_home, models_a="m-one@128k,#m-two@256k")
    cfg = load_config()
    assert cfg.profile("node-a:m-two") is None


def test_env_stray_vars_warn_by_name_only(cfg_home, capsys):
    """.env 混进非 key 变量（会静默压住 .env.example 的更新/自动下线）⇒
    重载配置时在 stderr 点名提醒；只报变量名，绝不回显任何值。"""
    _write_layers(cfg_home, env_lines=("NODE_A_KEY=sk-secret-do-not-print",
                                       "NODE_A_MODELS=evil@1"))
    load_config()
    err = capsys.readouterr().err
    assert "分层提醒" in err and "NODE_A_MODELS" in err
    assert "sk-secret-do-not-print" not in err, "告警不得回显 .env 里的任何值"
    # 纯 key 的 .env 不该触发提醒
    (cfg_home / ".env").write_text("NODE_A_KEY=sk-real\n", encoding="utf-8")
    os.utime(cfg_home / ".env", (time.time() + 5, time.time() + 5))
    load_config(force=True)
    assert "分层提醒" not in capsys.readouterr().err


def test_real_env_example_never_carries_key_values():
    """守卫分层契约：随仓库维护（AI 可改、进 git）的 .env.example 里，
    *_KEY 行只能是空占位——真实 key 只允许进 .env（AI 不可修改区）。"""
    real_example = Path(ac.__file__).resolve().parent.parent / ".env.example"
    lines = [ln.strip() for ln in real_example.read_text(encoding="utf-8").splitlines()]
    key_lines = [ln for ln in lines if "=" in ln and not ln.startswith("#")
                 and ln.split("=", 1)[0].strip().endswith("_KEY")]
    assert key_lines, ".env.example 应保留 *_KEY= 空占位行（契约测试核对变量名用）"
    for ln in key_lines:
        assert ln.split("=", 1)[1].strip() == "", f"git 维护层出现疑似真实 key：{ln.split('=')[0]} 行有值"


# ---------------- 2) ModelHealth 单元 ----------------


def _health(tmp_path):
    return ModelHealth(tmp_path / "data" / "model_health.json")


def test_failure_marks_today_only(tmp_path):
    h = _health(tmp_path)
    assert not h.quarantined("w")
    h.record_failure("w", "m-w", "")
    assert h.quarantined("w"), "一次终态失败 → 当天隔离"
    # 只有历史日期（非今天）不触发当天隔离
    h.record_failure("w2", "m-w2", "", day=_day(-1))
    assert not h.quarantined("w2")
    # 隔离不挡别的模型
    assert not h.quarantined("other")


def test_store_persists_across_instances(tmp_path):
    h = _health(tmp_path)
    h.record_failure("w", "m-w", "")
    h2 = _health(tmp_path)  # 新实例（=新进程）重新读盘
    assert h2.quarantined("w"), "当日隔离必须持久化，重启进程也不能再打坏节点"


def test_corrupt_store_file_recovers(tmp_path):
    store = tmp_path / "data" / "model_health.json"
    store.parent.mkdir(parents=True)
    store.write_text("{ not json", encoding="utf-8")
    h = ModelHealth(store)
    assert not h.quarantined("w")
    h.record_failure("w", "m-w", "")  # 从空白账本继续记，不抛
    assert h.quarantined("w")


# ---------------- 3) WorkerPool 集成（派工路径真的被隔离） ----------------


def _pool(health: ModelHealth, models: str = "m1,m2") -> WorkerPool:
    cfg = Config(_interpolate({
        "agents": [{"name": "st", "base_url": "u", "api_key": "k", "models": models}],
    }))
    return WorkerPool(cfg, exclude="st:m1", health=health)


def test_quarantined_worker_skipped_everywhere(tmp_path):
    h = _health(tmp_path)
    h.record_failure("st:m2", "m2", "")
    pool = _pool(h)
    assert "隔离" in pool.ask("st:m2", "任务A")
    assert pool.collect(["st:m2"], "任务A") == []
    many = pool.ask_many(["st:m2"], "任务A")
    assert "错误" in many and "st:m2" in many
    assert pool.vote("任务A", workers=["st:m2"]).startswith("错误"), (
        "候选收集全被隔离 → 直接报无可用方案，而不是打出真请求"
    )
    # 工人清单一目了然：隔离的工人被标注，主智能体选工时提前避开
    ov = pool.overview()
    assert "st:m2" in ov and "隔离" in ov


class _Boom:
    """假 LLMClient：任何调用都终态失败（重试后仍败出 LLMError）。"""

    def __init__(self, *_a, model: str = "", **_k):
        self.model = model  # 真 LLMClient 也暴露该属性，主智能体记档要用
        self.closed = 0

    def chat(self, messages, tools=None):
        raise LLMError("api", "boom", retryable=False)

    def close(self):
        self.closed += 1


def test_worker_final_failure_quarantines_it_for_today(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "LLMClient", _Boom)
    pool = _pool(_health(tmp_path))
    out = pool.ask("st:m2", "任务A")
    assert out.startswith("错误"), "失败要如实带回"
    assert pool.health.quarantined("st:m2"), "终态失败必须打上当日标记"
    assert "隔离" in pool.ask("st:m2", "任务B"), "当天后续派工直接跳过该节点"
    # 隔离只落在失败的那个模型上
    assert not pool.health.quarantined("st:m1")


class _Transient:
    """假 LLMClient：任何调用都抛**可重试**错误（模拟免费节点超时/限流/连接抖动）。"""

    def __init__(self, *_a, model: str = "", **_k):
        self.model = model
        self.closed = 0

    def chat(self, messages, tools=None):
        raise LLMError("timeout", "boom", retryable=True)

    def close(self):
        self.closed += 1


def test_transient_failure_cools_down_but_not_quarantined_today(tmp_path, monkeypatch):
    """可重试的瞬时失败只进短时冷却，绝不升级成全天封禁（#1/#2 的核心修复）。"""
    monkeypatch.setattr(orch, "LLMClient", _Transient)
    pool = _pool(_health(tmp_path))
    pool._cooldown_threshold = 1  # 一次失败即冷却，方便断言
    pool._cooldown_base = 5.0
    out = pool.ask("st:m2", "任务A")
    assert out.startswith("错误"), "失败要如实带回"
    assert not pool.health.quarantined("st:m2"), "瞬时失败不该当天隔离——明天/冷却后仍能回来"
    assert pool._is_in_cooldown("st:m2"), "但应进入短时冷却，暂时别反复打它"
    # 冷却只挡住当前这一下：到期后（这里手动清冷却）工人立刻恢复可派，且不报"隔离"
    pool._cooldowns.pop("st:m2", None)
    assert "隔离" not in pool.ask("st:m2", "任务B")
    # 终态失败模型不受影响：另一个模型的隔离判定独立
    assert not pool.health.quarantined("st:m1")
