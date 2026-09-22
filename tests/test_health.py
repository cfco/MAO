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

import json
import os
from datetime import date, timedelta

import pytest

import agent.config as ac
import agent.core.orchestrator as orch
import agent.core.session as sess_mod
from agent.config import Config, _interpolate, load_config
from agent.core.agent import Agent
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


# ---------------- 2) ModelHealth 单元 ----------------


def _health(tmp_path, retire_days=7):
    return ModelHealth(
        tmp_path / "data" / "model_health.json",
        tmp_path / ".env.example",
        retire_days=retire_days,
    )


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


def test_weekend_gap_still_retires(tmp_path):
    """运行日语义的核心诉求：隔了整个周末（自然日断档）也不打断连击。

    周一、周二各失败一次后周末没开机；下周一再失败 → 最近 3 个运行日
    （周一/周二/今天）全失败 → 下线。旧"自然日连续"实现会在断档处清零。
    """
    h = _health(tmp_path, retire_days=3)
    assert h.record_failure("w", "m-w", "", day=_day(-8)) is None  # 上上周六？不重要，重要的是运行日
    assert h.record_failure("w", "m-w", "", day=_day(-7)) is None
    assert h.record_failure("w", "m-w", "") is not None  # 今天：第 3 个运行日，触发下线


def test_active_day_without_failure_breaks_streak(tmp_path):
    """打断连击的只能是"开过机但该模型没失败"的运行日（比如当天它成功了）。"""
    h = _health(tmp_path, retire_days=3)
    assert h.record_failure("w", "m-w", "", day=_day(-8)) is None
    assert h.record_failure("w", "m-w", "", day=_day(-7)) is None
    h.note_active(_day(-1))  # 昨天开过程序、这个模型没失败 → 窗口被它撑破
    assert h.record_failure("w", "m-w", "") is None, "存在没失败的运行日 → 连击重新起算"


def test_streak_below_active_day_count_does_not_retire(tmp_path):
    """运行日样本不足 retire_days 个时不判下线（刚装好程序不该当天就下线模型）。"""
    h = _health(tmp_path, retire_days=3)
    assert h.record_failure("w", "m-w", "", day=_day(-1)) is None
    assert h.record_failure("w", "m-w", "") is None, "只有 2 个运行日样本，不足 3"


def test_v1_store_reads_and_restart_streak(tmp_path):
    """旧版档案（无 active_dates）可读：隔离照旧，连击从当天重新积累。"""
    store = tmp_path / "data" / "model_health.json"
    store.parent.mkdir(parents=True)
    store.write_text(
        json.dumps({"version": 1, "models": {"w": {"fail_dates": [_day(-2), _day(-1)]}}}),
        encoding="utf-8",
    )
    h = ModelHealth(store, None, retire_days=2)
    assert h.record_failure("w", "m-w", "") is None, "历史运行日未知，不该直接判下线"
    text = store.read_text(encoding="utf-8")
    assert '"version": 2' in text and _day(-2) in text, "回写升级为 v2 且保留历史失败记录"


def test_retire_rewrites_env_example(tmp_path):
    (tmp_path / ".env.example").write_text(
        "NODE_A_MODELS=m-one@128k,m-two@256k,#m-three@64k\n"
        "NODE_B_MODELS=m-two@512k,other@256k\n"
        "LLM_MODEL=m-one\n",
        encoding="utf-8",
    )
    h = _health(tmp_path, retire_days=3)
    assert h.record_failure("node-a:m-two", "m-two", "NODE_A_MODELS", day=_day(-2)) is None
    assert h.record_failure("node-a:m-two", "m-two", "NODE_A_MODELS", day=_day(-1)) is None
    notice = h.record_failure("node-a:m-two", "m-two", "NODE_A_MODELS")
    assert notice and "下线" in notice
    text = (tmp_path / ".env.example").read_text(encoding="utf-8")
    lines = dict(kv.split("=", 1) for kv in text.splitlines() if kv)
    # 命中的模型加 #；同模型多站一起下线；未命中的条目与 LLM_MODEL 兜底行原样
    assert lines["NODE_A_MODELS"] == "m-one@128k,#m-two@256k,#m-three@64k"
    assert lines["NODE_B_MODELS"] == "#m-two@512k,other@256k"
    assert lines["LLM_MODEL"] == "m-one"
    # 已 disabled 后继续失败：不再重复提示、不再重复加 #
    assert h.record_failure("node-a:m-two", "m-two", "NODE_A_MODELS") is None
    text2 = (tmp_path / ".env.example").read_text(encoding="utf-8")
    assert text2 == text


def test_store_persists_across_instances(tmp_path):
    h = _health(tmp_path)
    h.record_failure("w", "m-w", "")
    h2 = _health(tmp_path)  # 新实例（=新进程）重新读盘
    assert h2.quarantined("w"), "当日隔离必须持久化，重启进程也不能再打坏节点"


def test_corrupt_store_file_recovers(tmp_path):
    store = tmp_path / "data" / "model_health.json"
    store.parent.mkdir(parents=True)
    store.write_text("{ not json", encoding="utf-8")
    h = ModelHealth(store, None)
    assert not h.quarantined("w")
    h.record_failure("w", "m-w", "")  # 从空白账本继续记，不抛
    assert h.quarantined("w")


def test_retire_without_env_file_reports_manual(tmp_path):
    h = ModelHealth(tmp_path / "h.json", None, retire_days=1)
    notice = h.record_failure("w", "m-w", "")
    assert notice and "手动" in notice, "无法自动改写时要如实提示人工确认"
    # 档案里仍记了 disabled，不会每天重复打网络
    assert h.record_failure("w", "m-w", "") is None


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


def test_cache_hit_bypasses_quarantine(tmp_path):
    h = _health(tmp_path)
    h.record_failure("st:m2", "m2", "")
    pool = _pool(h)
    pool._cache[("st:m2", "任务A", None)] = "cached-answer"
    assert pool.ask("st:m2", "任务A") == "cached-answer", "缓存命中零成本，不该被隔离挡住"
    assert "隔离" in pool.ask("st:m2", "任务B"), "未缓存的任务仍要走隔离"


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


def test_master_failure_recorded_not_blocked(tmp_path, monkeypatch):
    """主智能体失败：记档（参与连击计数），但不拦截主的下一轮——主是用户选的。"""
    monkeypatch.setattr(sess_mod, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(sess_mod, "TRASH_DIR", tmp_path / "trash")
    monkeypatch.setattr(ac, "ROOT", tmp_path)  # 健康档案/清单都落在 tmp，不碰真仓库
    monkeypatch.setattr("agent.core.agent.LLMClient", _Boom)
    cfg = Config(_interpolate({
        "agents": [],
        "llm": {"base_url": "u", "api_key": "k", "model": "solo-m"},
        "session": {"flush_batch": 1, "flush_interval": 0.0},
    }))
    bot = Agent(cfg, profile=None, enable_workers=False)
    out = bot.run("hello")
    assert "模型调用失败" in out
    snap = bot.health.snapshot()
    entry = snap.get("solo:solo-m")
    assert entry and _today() in entry["fail_dates"], "主失败也要记入健康档案"
    # 未达连击阈值 → 不标 #、不隔离下一轮（再跑一轮仍是正常报错路径）
    out2 = bot.run("hello again")
    assert "模型调用失败" in out2
    bot.close()
