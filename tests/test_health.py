"""模型健康 + 配置分层的回归测试：全部离线，无需 API key / 网络。

覆盖两块（2026-09-23 配置重构二期）：
1) 配置分层：model_registry.txt 承载模型清单（基础层，随仓库维护，自动下线改写它），
   .env 承载全部配置值（端点 / KEY / 参数，覆盖层）；优先级 shell > .env > registry；
   registry / .env 变化都要让配置缓存失效；.env 里混进模型清单变量要点名提醒。
2) 模型健康：一次终态失败 → 当日隔离不再派工（缓存命中不受影响）；
   连续 retire_days 个"运行日"（程序实际启动过的天，未运行的日子不计入也不打断）
   都失败 → 自动在 model_registry.txt 对应行给模型标 # 下线
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
    monkeypatch.setattr(ac, "_cfg_cache_registry_mtime", 0.0)
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
    """写三份文件：config.yaml 骨架 + model_registry.txt（模型名）+ .env（配置值）。"""
    (home / "config.yaml").write_text(CONFIG_YAML, encoding="utf-8")
    (home / "model_registry.txt").write_text(
        "# 分层基础：模型清单\n"
        f"NODE_A_MODELS={models_a}\n"
        "LLM_MODEL=m-one\n",
        encoding="utf-8",
    )
    # .env 默认只给 key；需要端点时由调用方自提（ENDPOINT 属配置层）：
    (home / ".env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")


# ---------------- 1) 配置分层（registry + .env） ----------------


def test_layers_registry_base_env_override(cfg_home):
    """模型名在 model_registry.txt、配置值（端点/key）在 .env：池应能建起来。"""
    _write_layers(cfg_home, env_lines=("NODE_A_ENDPOINT=https://a.example/v1",
                                       "NODE_A_KEY=sk-real"))
    cfg = load_config()
    assert cfg.agent_profiles, "registry 提供模型名、.env 提供端点/key，池应能建起来"
    p = cfg.profile("node-a:m-one")
    assert p is not None and p.base_url == "https://a.example/v1"
    assert p.api_key == "sk-real", ".env 的 key 必须生效"
    assert p.models_env == "NODE_A_MODELS", "要能反查模型清单所在变量（健康下线改写用）"
    assert cfg.llm.get("model") == "m-one", "LLM_MODEL 由 registry 提供"
    assert cfg.llm.get("api_key") == "", "LLM_API_KEY 未在 .env 配置 → 插值为空串"
    assert cfg.retire_days == 7, "collaboration.retire_days 缺省 7 个运行日"


def test_shell_env_wins_over_both_layers(cfg_home, monkeypatch):
    """shell 预置变量优先级最高：既盖 .env（key/端点），也盖 registry（模型清单）。"""
    monkeypatch.setenv("NODE_A_ENDPOINT", "http://shell.wins/v1")
    monkeypatch.setenv("NODE_A_MODELS", "shell-only@256k")
    monkeypatch.setenv("NODE_A_KEY", "sk-shell")
    _write_layers(cfg_home, env_lines=("NODE_A_ENDPOINT=http://dotenv.lose/v1",
                                       "NODE_A_KEY=sk-real"))
    cfg = load_config()
    # 只剩 1 个模型 → 名字归一化为站名 node-a（不带 :模型 后缀）；model 只留裸名
    p = cfg.profile("node-a")
    assert p is not None and p.model == "shell-only", "shell 的模型清单必须盖掉 registry"
    assert p.base_url == "http://shell.wins/v1"
    assert p.api_key == "sk-shell", "shell 已导出的 key 不被 .env 覆盖"
    assert cfg.profile("node-a:m-one") is None, "registry 的 m-one 不在 shell 清单里"


def test_missing_env_file_still_loads_from_registry(cfg_home):
    _write_layers(cfg_home)
    (cfg_home / ".env").unlink()
    cfg = load_config()
    p = cfg.profile("node-a:m-one")
    assert p is not None and p.api_key == "", "没有 .env 时池照常构建（registry 管模型名），key 为空占位"


def test_registry_change_invalidates_cache(cfg_home):
    _write_layers(cfg_home)
    cfg1 = load_config()
    assert load_config() is cfg1, "三份文件都没动时应命中缓存"
    # 只改 model_registry.txt（比如 git pull 下线了 m-two），并显式推后 mtime
    reg = cfg_home / "model_registry.txt"
    reg.write_text(
        reg.read_text(encoding="utf-8").replace("NODE_A_MODELS=m-one@128k,m-two@256k",
                                                "NODE_A_MODELS=m-one@128k"),
        encoding="utf-8",
    )
    st = reg.stat()
    os.utime(reg, (st.st_atime + 5, st.st_mtime + 5))
    cfg2 = load_config()
    assert cfg2.profile("node-a:m-two") is None, "registry 变化必须让缓存失效重载"
    # 删剩单模型后按规则改名为站名（不再带 :模型 后缀）
    assert cfg2.profile("node-a") is not None


def test_hash_blocked_model_untouched_by_layering(cfg_home):
    _write_layers(cfg_home, models_a="m-one@128k,#m-two@256k")
    cfg = load_config()
    assert cfg.profile("node-a:m-two") is None


def test_env_stray_models_vars_warn_by_name_only(cfg_home, capsys):
    """.env 混进模型清单变量（会静默压住 model_registry.txt 的 git 更新/自动下线）⇒
    重载配置时在 stderr 点名提醒；只报变量名，绝不回显任何值。"""
    _write_layers(cfg_home, env_lines=("NODE_A_MODELS=evil@1",
                                       "NODE_A_KEY=sk-secret-do-not-print"))
    load_config()
    err = capsys.readouterr().err
    assert "分层提醒" in err and "NODE_A_MODELS" in err
    assert "sk-secret-do-not-print" not in err, "告警不得回显 .env 里的任何值"
    # 纯配置的 .env（端点/key/参数）不该触发提醒
    (cfg_home / ".env").write_text(
        "NODE_A_ENDPOINT=https://a.example/v1\nNODE_A_KEY=sk-real\nCOLLAB_RETIRE_DAYS=3\n",
        encoding="utf-8",
    )
    os.utime(cfg_home / ".env", (time.time() + 5, time.time() + 5))
    load_config(force=True)
    assert "分层提醒" not in capsys.readouterr().err


def test_inline_comment_stripped_in_dotenv(cfg_home):
    """.env 值里"空格+#"的行内注释要剥掉——否则数字/端点参数会被注释文本污染成非法值。"""
    _write_layers(cfg_home, env_lines=("NODE_A_ENDPOINT=https://a.example/v1 # 端点",
                                       "NODE_A_KEY=sk-real",
                                       "COLLAB_RETIRE_DAYS=7   # 连续 7 个运行日"))
    cfg = load_config()
    assert cfg.retire_days == 7, "COLLAB_RETIRE_DAYS 的行内注释应被剥离"
    p = cfg.profile("node-a:m-one")
    assert p is not None and p.base_url == "https://a.example/v1", "端点的行内注释应被剥离"


def test_real_registry_never_carries_key_values():
    """守卫契约：随仓库维护（AI 可改、进 git）的 model_registry.txt 里不得出现
    *_KEY 行——真实 key 只允许进 .env（保密、不进 git）。"""
    real_registry = Path(ac.__file__).resolve().parent.parent / "model_registry.txt"
    lines = [ln.strip() for ln in real_registry.read_text(encoding="utf-8").splitlines()]
    key_lines = [ln for ln in lines if "=" in ln and not ln.startswith("#")
                 and ln.split("=", 1)[0].strip().endswith("_KEY")]
    assert not key_lines, f"model_registry.txt 出现疑似 key 变量行：{key_lines}"


# ---------------- 2) ModelHealth 单元 ----------------

REGISTRY_TXT = "NODE_A_MODELS=m-one@128k,m-two@256k\nLLM_MODEL=m-one\n"


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


# ---------------- 3) 自动下线：连续运行日失败 → registry 加 # ----------------

def _registry(tmp_path) -> Path:
    reg = tmp_path / "model_registry.txt"
    reg.write_text(REGISTRY_TXT, encoding="utf-8")
    return reg


def _retiring(tmp_path, retire_days=7):
    return ModelHealth(tmp_path / "h.json", _registry(tmp_path), retire_days=retire_days)


def test_retire_marks_model_in_registry_after_retire_days(tmp_path):
    """连续 retire_days 个运行日都失败 → 在 model_registry.txt 给该模型加 #。"""
    h = _retiring(tmp_path, retire_days=3)
    for i in range(3):
        h.record_failure("st:m-one", "m-one", "NODE_A_MODELS", day=_day(i - 3))
    text = (tmp_path / "model_registry.txt").read_text(encoding="utf-8")
    assert "#m-one" in text, f"达标后模型必须被标 # 下线：\n{text}"
    assert "#m-two" not in text, "其它模型不受牵连（m-two 原样无 #）"
    # 档案里记了 disabled 标记（防二次改写下线提醒重复刷屏）
    assert h.snapshot()["st:m-one"].get("disabled") == _day(-1)


def test_retire_does_not_fire_below_retire_days(tmp_path):
    h = _retiring(tmp_path, retire_days=3)
    for i in range(2):  # 只失败 2 个运行日
        h.record_failure("st:m-one", "m-one", "NODE_A_MODELS", day=_day(i - 2))
    text = (tmp_path / "model_registry.txt").read_text(encoding="utf-8")
    assert "#m-one" not in text, "运行日样本不足时不许提前下线"


def test_unrun_days_not_counted_nor_interrupt(tmp_path):
    """「没运行项目的天」既不计入、也不打断连击（用户核心诉求）。"""
    h = _retiring(tmp_path, retire_days=3)
    # 运行日 = 第 1、3、4 天；第 2 天没开机（不在 active_dates），不打断窗口
    for d in (_day(-4), _day(-2), _day(-1)):
        h.record_failure("st:m-one", "m-one", "NODE_A_MODELS", day=d)
    text = (tmp_path / "model_registry.txt").read_text(encoding="utf-8")
    assert "#m-one" in text, "缺一天没运行不影响：最近 3 个运行日仍全失败 → 下线"


def test_success_running_day_breaks_streak(tmp_path):
    """模型在某运行日没失败（成功或未被调用）→ 该日不算失败日，不得误杀。"""
    h = _retiring(tmp_path, retire_days=3)
    h.record_failure("st:m-one", "m-one", "NODE_A_MODELS", day=_day(-3))
    h.note_active(_day(-2))  # 这个运行日模型没失败
    h.record_failure("st:m-one", "m-one", "NODE_A_MODELS", day=_day(-1))
    text = (tmp_path / "model_registry.txt").read_text(encoding="utf-8")
    assert "#m-one" not in text, "存在『未失败』的运行日 → 不满足全失败，不许下线"


def test_retire_is_idempotent_and_informs_llm_model_untouched(tmp_path):
    """已 # 下线的模型再失败：不重复改写，仍报提示；LLM_MODEL 兜底行绝不被自动动。"""
    reg = _registry(tmp_path)
    h = ModelHealth(tmp_path / "h.json", reg, retire_days=2)
    for i in range(2):
        h.record_failure("st:m-one", "m-one", "NODE_A_MODELS", day=_day(i - 2))
    text = reg.read_text(encoding="utf-8")
    assert "#m-one" in text and "LLM_MODEL=m-one" in text, "LLM_MODEL 行原样保留"
    # 再模拟一次失败：disabled 标记在案 → 不再重复改写 registry、也不再刷下线提示（幂等）
    h2 = ModelHealth(tmp_path / "h.json", reg, retire_days=2)
    notice = h2.record_failure("st:m-one", "m-one", "NODE_A_MODELS", day=_day(-1))
    assert notice is None, "已下线模型再失败不重复提示/改写（防刷屏），实际返回 None"
    assert reg.read_text(encoding="utf-8") == text, "幂等：registry 内容不再变化"


# ---------------- 4) WorkerPool 集成（派工路径真的被隔离） ----------------

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
    v = pool.vote("任务A", workers=["st:m2"])
    assert v["ok"] is False and v["report"].startswith("错误"), (
        "候选收集全被隔离 → 直接报无可用方案，而不是打出真请求"
    )
    # 隔离态在预检快照里一目了然：主智能体派工前查 health 即可避开，不用挨个撞错误提示
    hs = {x["worker"]: x for x in pool.health_status()}
    assert hs["st:m2"]["quarantined_today"] is True and hs["st:m2"]["available"] is False


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
