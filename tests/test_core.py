"""MAO 核心逻辑测试：全部离线，无需 API key / 网络。

覆盖（历史遗留逻辑补测试 + 本次审查后的回归保护）：
- 工具注册表：重复注册、白名单裁剪、未知工具/坏 JSON 容错、schema 导出
- 配置解析：环境变量插值、一站多模型展开、屏蔽(#)、brief 不含 key、profile 查找
- LLMClient：指数退避边界、可重试错误退避重试、认证错误不重试
- 会话落盘限流：内存即时更新 + 缓冲合并写盘 + flush 兜底 + 往返持久化
- WorkerPool：节点冷却逻辑、ask_many 无可用工人、vote 空池、未知工人
- ask 并发去抖：同 key 并发只打一次网络，跟随者复用结果
- ask 在飞去重兜底：发起者挂死时跟随者超时接管，不无限阻塞
- ask system 归一：None/""/前后空白 → 同一缓存键，命中缓存不重复打网络
- events.to_event：error 类型显式归一为 {"event":"error","message":...}
- 会话落盘限流：环境变量（CLI 覆盖）优先于 config.yaml
- 本轮审查修正的回归：命令黑名单按「命令首词」匹配（ruff format / make clean
  不得被误拦，& 串接的危险命令仍须拦下）、-EncodedCommand 缩写、空 choices 归类为
  可重试错误、session_id 与 skill 脚本名的路径穿越、在飞接管不得清掉新发起者的
  inflight、空回复不静默
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openai import AuthenticationError, RateLimitError

import agent.core.orchestrator as orch
import agent.core.session as sess_mod
from agent.config import Config, _interpolate, _split_models
from agent.core.events import to_event
from agent.core.llm import (
    ERR_API,
    ERR_AUTH,
    ERR_RATE_LIMIT,
    LLMClient,
    LLMError,
    _retryable_from_type,
)
from agent.core.orchestrator import WorkerPool
from agent.core.session import Session
from agent.tools.base import FunctionTool, ToolRegistry

# ---------------- 工具注册表 ----------------

def test_registry_duplicate_raises():
    reg = ToolRegistry()
    reg.register(FunctionTool("t", "d", {}, lambda a: "x"))
    with pytest.raises(ValueError):
        reg.register(FunctionTool("t", "d", {}, lambda a: "y"))


def test_registry_restrict_returns_removed():
    reg = ToolRegistry()
    reg.register(FunctionTool("a", "d", {}, lambda a: "1"))
    reg.register(FunctionTool("b", "d", {}, lambda a: "2"))
    removed = reg.restrict(["a"])
    assert removed == ["b"]
    assert reg.get("b") is None
    assert reg.get("a") is not None


def test_registry_execute_unknown_and_bad_json():
    reg = ToolRegistry()
    reg.register(FunctionTool("t", "d", {}, lambda a: "ok"))
    assert "未知工具" in reg.execute("nope", "{}")
    # 坏 JSON 参数应转成错误文本而非抛异常
    assert "合法 JSON" in reg.execute("t", "not-json{")
    # 正常调用
    assert reg.execute("t", "{}") == "ok"
    # schema 导出数量与注册一致
    assert len(reg.openai_schemas()) == 1


# ---------------- 配置解析 ----------------

def test_interpolate_env(monkeypatch):
    monkeypatch.setenv("MAO_TEST_TOKEN", "SECRET")
    assert _interpolate("x ${MAO_TEST_TOKEN} y") == "x SECRET y"
    # 缺失变量替换为空串
    assert _interpolate("a${NOPE_VAR}b") == "ab"


def test_split_models_shield_and_multi():
    # 返回 (模型名, 上下文长度) 元组；# 前缀屏蔽
    assert _split_models("a,#b,c") == [("a", 0), ("c", 0)]
    assert _split_models(["x", "#y"]) == [("x", 0)]
    assert _split_models("") == []
    # @长度标注：k/K 按 1024 换算
    assert _split_models("gpt@128k") == [("gpt", 128 * 1024)]
    assert _split_models(["m1", "m2"]) == [("m1", 0), ("m2", 0)]


def test_config_multimodel_expansion_and_lookup():
    data = {"agents": [{"name": "st", "base_url": "u", "api_key": "k", "models": ["m1", "m2"]}]}
    cfg = Config(_interpolate(data))
    names = {p.name for p in cfg.agent_profiles}
    assert names == {"st:m1", "st:m2"}
    assert all(p.api_key == "k" for p in cfg.agent_profiles)
    # 一站多模型：上下文长度随元组拆出
    m1 = cfg.profile("st:m1")
    assert m1 is not None and m1.model == "m1" and m1.context_length == 0
    # 单模型保留站名
    cfg2 = Config(_interpolate({"agents": [{"name": "solo", "base_url": "u", "api_key": "k", "model": "m"}]}))
    assert cfg2.agent_profiles[0].name == "solo"
    # profile 查找
    assert cfg.profile("st:m1") is not None
    assert cfg.profile("st:m9") is None
    # brief 不含 key
    assert "api_key" not in cfg.profile("st:m1").brief()


def test_config_session_defaults_and_override():
    # 默认（无 session 段）
    cfg = Config(_interpolate({"agents": []}))
    assert cfg.session_flush_batch == 16
    assert cfg.session_flush_interval == 2.0
    # 覆盖
    cfg2 = Config(_interpolate({"session": {"flush_batch": 4, "flush_interval": 5.0}}))
    assert cfg2.session_flush_batch == 4
    assert cfg2.session_flush_interval == 5.0
    # 下限保护（batch 至少 1，interval 至少 0.0）
    cfg3 = Config(_interpolate({"session": {"flush_batch": 0, "flush_interval": -1}}))
    assert cfg3.session_flush_batch == 1
    assert cfg3.session_flush_interval == 0.0


# ---------------- LLMClient 重试 / 退避 ----------------

def _fake_response(content: str):
    msg = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _fake_openai_response(status: int):
    """构造 openai 异常所需的伪 response（openai 3.x 会访问 response.request 等属性）。"""
    return SimpleNamespace(
        request=SimpleNamespace(method="GET", url="http://x"),
        status_code=status,
        headers={},
    )


def test_llm_backoff_bounds():
    from agent.core.llm import _MAX_BACKOFF

    b_rate = LLMClient._backoff(0, ERR_RATE_LIMIT)
    b_other = LLMClient._backoff(0, "other")
    assert 0 <= b_rate <= _MAX_BACKOFF + 1
    assert 0 <= b_other <= _MAX_BACKOFF + 1
    # 限流退避基数应大于其它错误
    assert b_rate > b_other
    # 退避随 attempt 非递减（封顶内）
    assert LLMClient._backoff(1, ERR_RATE_LIMIT) >= b_rate


def test_llm_retry_then_success(monkeypatch):
    client = LLMClient("http://x", "k", "m", max_retries=2)
    monkeypatch.setattr(client, "_backoff", lambda *a, **k: 0)  # 测试不真睡
    state = {"n": 0}

    def create(**_kwargs):
        state["n"] += 1
        if state["n"] <= 2:
            # 限流(429) = 可重试错误，验证 chat 的退避重试编排
            raise RateLimitError("rl", response=_fake_openai_response(429), body=None)
        return _fake_response("hi")

    fake = MagicMock()
    fake.chat.completions.create.side_effect = create
    client.client = fake

    out = client.chat([{"role": "user", "content": "q"}])
    assert out["content"] == "hi"
    assert state["n"] == 3  # 失败2次 + 成功1次


def test_llm_auth_not_retried(monkeypatch):
    client = LLMClient("http://x", "k", "m", max_retries=2)
    state = {"n": 0}

    def create(**_kwargs):
        state["n"] += 1
        # 认证失败 = 不可重试错误，验证 chat 立即判失败不重试
        raise AuthenticationError("bad", response=_fake_openai_response(401), body=None)

    fake = MagicMock()
    fake.chat.completions.create.side_effect = create
    client.client = fake

    with pytest.raises(LLMError) as exc:
        client.chat([{"role": "user", "content": "q"}])
    assert exc.value.error_type == ERR_AUTH
    assert exc.value.retryable is False
    assert state["n"] == 1  # 认证失败立即判失败，不重试


def test_retryable_from_type():
    assert _retryable_from_type(ERR_RATE_LIMIT) is True
    assert _retryable_from_type(ERR_AUTH) is False


# ---------------- 会话落盘限流 ----------------

def test_session_in_memory_immediate():
    s = Session("mem-1")
    s.add("user", "hello")
    # 内存历史立即更新（供主循环拼装）
    assert s.history[-1] == {"role": "user", "content": "hello"}


def test_session_flush_coalesces(tmp_path, monkeypatch):
    monkeypatch.setattr(sess_mod, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(sess_mod, "TRASH_DIR", tmp_path / "trash")
    monkeypatch.setattr(sess_mod, "FLUSH_BATCH", 5)
    monkeypatch.setattr(sess_mod, "FLUSH_INTERVAL", 1000)  # 仅 batch 触发，便于断言

    s = Session("s1")
    for i in range(3):
        s.add("user", f"m{i}")
    # 未达 batch：仍在内存缓冲，未写盘
    assert len(s._pending) == 3
    assert not s._path.exists() or s._path.read_text(encoding="utf-8").count("\n") == 0
    # 强制落盘
    s.flush()
    assert s._path.read_text(encoding="utf-8").count("\n") == 3
    assert len(s._pending) == 0
    # 往返持久化：新实例从磁盘恢复
    s2 = Session("s1")
    assert len(s2.history) == 3


def test_session_auto_flush_on_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(sess_mod, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(sess_mod, "TRASH_DIR", tmp_path / "trash")
    monkeypatch.setattr(sess_mod, "FLUSH_BATCH", 3)
    monkeypatch.setattr(sess_mod, "FLUSH_INTERVAL", 1000)

    s = Session("s2")
    for _ in range(3):
        s.add("user", "x")
    # 达到 batch 阈值自动落盘
    assert len(s._pending) == 0
    assert s._path.read_text(encoding="utf-8").count("\n") == 3


def test_session_explicit_flush_params(tmp_path, monkeypatch):
    monkeypatch.setattr(sess_mod, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(sess_mod, "TRASH_DIR", tmp_path / "trash")

    # 显式传参应覆盖模块常量（不依赖 monkeypatch 常量），batch=3 触发落盘
    s = Session("exp", flush_batch=3, flush_interval=1000)
    for _ in range(3):
        s.add("user", "x")
    assert len(s._pending) == 0  # 达到显式 batch 已落盘
    assert s._path.read_text(encoding="utf-8").count("\n") == 3

    # 显式 interval=0 应每次 add 即落盘（低频但强制）
    s2 = Session("exp2", flush_batch=999, flush_interval=0.0)
    s2.add("user", "y")
    assert len(s2._pending) == 0
    assert s2._path.read_text(encoding="utf-8").count("\n") == 1


# ---------------- WorkerPool 编排逻辑 ----------------

def test_worker_pool_cooldown_logic():
    cfg = Config(_interpolate({"agents": []}))
    pool = WorkerPool(cfg, exclude=None)
    pool._cooldown_threshold = 1
    pool._cooldown_base = 1.0
    name = "node-x"
    pool._record_result(name, ok=False)
    assert pool._is_in_cooldown(name) is True
    pool._record_result(name, ok=True)
    assert pool._is_in_cooldown(name) is False


def test_ask_unknown_worker():
    cfg = Config(_interpolate({"agents": []}))
    pool = WorkerPool(cfg, exclude=None)
    assert "不存在" in pool.ask("ghost", "hi")


def test_ask_many_no_valid_workers():
    cfg = Config(_interpolate({"agents": []}))
    pool = WorkerPool(cfg, exclude=None)
    out = pool.ask_many(["nope1", "nope2"], "task")
    assert "没有可用的子智能体" in out


def test_vote_empty_pool():
    cfg = Config(_interpolate({"agents": []}))
    pool = WorkerPool(cfg, exclude=None)
    assert pool.vote("q") == "错误：没有可投票的子智能体。"


# ---------------- ask 并发去抖（同 key 只打一次网络） ----------------

def test_ask_concurrent_dedup_same_key():
    # 用虚拟工人配置，保证池非空，使去抖路径必然被执行（不依赖真实 config.yaml）。
    # 确定性写法：跟随者任务在判定"是否跟随"前先等发起者已建立 inflight，
    # 消除"发起者瞬间完成→跟随者变新发起者"的时序竞态。
    cfg = Config(_interpolate({"agents": [{"name": "dummy", "base_url": "http://x", "api_key": "k", "model": "m"}]}))
    pool = WorkerPool(cfg, exclude=None)
    assert pool.names(), "虚拟工人应已进入池中"
    worker = pool.names()[0]

    calls = {"n": 0}
    attached = {"n": 0}
    followers = 8
    inflight_ready = threading.Event()
    release = threading.Event()  # 主线程确认所有跟随者 attach 后放行发起者

    class FakeClient:
        def chat(self, messages):
            inflight_ready.set()  # 发起者已建立 inflight
            calls["n"] += 1
            release.wait(timeout=5)  # 等全部跟随者就位后再返回，保证去抖命中
            return {"role": "assistant", "content": "ok"}

    real_ask = pool.ask

    def follower_ask():
        inflight_ready.wait(timeout=5)  # 确保发起者已建立 inflight 再判定
        key = (worker, "same-task", None)
        with pool._lock:
            is_follower = key in pool._cache or pool._inflight.get(key) is not None
        if is_follower:
            attached["n"] += 1
        return real_ask(worker, "same-task")

    orig = orch.LLMClient
    orch.LLMClient = lambda *a, **k: FakeClient()
    try:
        # 1) 发起者（走原始 ask）建立 inflight 并进入网络
        with ThreadPoolExecutor(max_workers=1) as ex_lead:
            lead = ex_lead.submit(real_ask, worker, "same-task")
            assert inflight_ready.wait(timeout=5)
        # 2) 并发跟随者：复用发起者结果，不重复打网络
        with ThreadPoolExecutor(max_workers=followers) as ex:
            futs = [ex.submit(follower_ask) for _ in range(followers)]
            deadline = time.time() + 5
            while attached["n"] < followers and time.time() < deadline:
                time.sleep(0.01)
            assert attached["n"] == followers, f"应全部 attach，实际 {attached['n']}"
            release.set()  # 放行发起者返回
            results = [f.result(timeout=10) for f in futs]
            lead_result = lead.result(timeout=10)
        assert calls["n"] == 1, f"并发同 key 应只打 1 次网络，实际 {calls['n']}"
        assert all(r == "ok" for r in results + [lead_result])
        # 3) 发起者成功后已入缓存：后续同 key 命中缓存，不再打网络
        assert real_ask(worker, "same-task") == "ok"
        assert calls["n"] == 1
    finally:
        orch.LLMClient = orig
        release.set()  # 兜底：无论如何放行发起者，避免线程挂起


def test_ask_distinct_keys_each_call():
    cfg = Config(_interpolate({"agents": [{"name": "dummy", "base_url": "http://x", "api_key": "k", "model": "m"}]}))
    pool = WorkerPool(cfg, exclude=None)
    assert pool.names(), "虚拟工人应已进入池中"
    worker = pool.names()[0]

    calls = {"n": 0}

    class FakeClient:
        def chat(self, messages):
            calls["n"] += 1
            return {"role": "assistant", "content": "ok"}

    orig = orch.LLMClient
    orch.LLMClient = lambda *a, **k: FakeClient()
    try:
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = [ex.submit(pool.ask, worker, f"distinct-{i}") for i in range(3)]
            for f in futs:
                assert f.result(timeout=10) == "ok"
        assert calls["n"] == 3  # 不同 key 各自打网络，不过度去重
    finally:
        orch.LLMClient = orig


def test_ask_concurrent_initiator_failure_shared_with_followers():
    # 专项：发起者网络失败（不止成功路径）。并发同 key 时只有发起者打网络，
    # 失败文本共享给跟随者（跟随者拿到错误文本而非异常），且失败不入缓存（后续重试）。
    # 关键：发起者和跟随者必须在发起者 inflight 存活期间（完成前）并发执行。
    cfg = Config(_interpolate({"agents": [{"name": "dummy", "base_url": "http://x", "api_key": "k", "model": "m"}]}))
    pool = WorkerPool(cfg, exclude=None)
    assert pool.names(), "虚拟工人应已进入池中"
    worker = pool.names()[0]

    calls = {"n": 0}
    attached = {"n": 0}
    followers = 4
    inflight_ready = threading.Event()
    release = threading.Event()

    class FailingClient:
        def chat(self, messages):
            inflight_ready.set()  # 发起者已建立 inflight
            calls["n"] += 1
            release.wait(timeout=5)  # 等全部跟随者就位后再抛错，保证去抖命中
            raise RuntimeError("network down")

    real_ask = pool.ask

    def follower_ask():
        inflight_ready.wait(timeout=5)  # 确保发起者已建立 inflight 再判定
        key = (worker, "boom", None)
        with pool._lock:
            is_follower = key in pool._cache or pool._inflight.get(key) is not None
        if is_follower:
            attached["n"] += 1
        return real_ask(worker, "boom")

    orig = orch.LLMClient
    orch.LLMClient = lambda *a, **k: FailingClient()
    try:
        # 发起者用后台线程启动，不阻塞主线程，保证 inflight 在发起者完成前一直存活
        lead_future = Future()

        def _lead():
            try:
                lead_future.set_result(real_ask(worker, "boom"))
            except Exception as exc:
                lead_future.set_exception(exc)

        threading.Thread(target=_lead, name="lead", daemon=True).start()
        assert inflight_ready.wait(timeout=5), "发起者应已建立 inflight 并进入网络段"

        # 并发跟随者：必须在发起者 inflight 存活期间进来 attach
        with ThreadPoolExecutor(max_workers=followers) as ex:
            futs = [ex.submit(follower_ask) for _ in range(followers)]
            deadline = time.time() + 5
            while attached["n"] < followers and time.time() < deadline:
                time.sleep(0.01)
            assert attached["n"] == followers, f"应全部 attach，实际 {attached['n']}"
            release.set()  # 放行发起者抛错
            results = [f.result(timeout=10) for f in futs]
            lead_result = lead_future.result(timeout=10)
        # 只有发起者真正打网络（失败也算一次），跟随者复用其失败文本
        assert calls["n"] == 1, f"并发同 key 失败应只打 1 次网络，实际 {calls['n']}"
        # 跟随者拿到的是失败文本（不是异常），且与发起者一致
        assert all(r.startswith("错误：") for r in results + [lead_result])
        assert all(r == lead_result for r in results)
        # 失败不入缓存；且终态失败打了当日隔离标记：同天再派同一工人
        # 直接快速失败，不再打网络（calls 保持 1）——两重语义一起验证。
        again = real_ask(worker, "boom")
        assert calls["n"] == 1, "当日隔离生效：不该再为该工人打网络"
        assert again.startswith("错误：") and "隔离" in again
    finally:
        orch.LLMClient = orig


def test_ask_inflight_timeout_follower_takes_over():
    # 在飞去重兜底：发起者挂死（远超 inflight_timeout）时，跟随者应在超时后接管、
    # 重新打网络并返回结果，而不是无限阻塞；且接管者确实重新发起了一次网络调用。
    # 用 collaboration.inflight_timeout=0.3 让兜底快速触发，避免测试久等。
    cfg = Config(_interpolate({
        "agents": [{"name": "dummy", "base_url": "http://x", "api_key": "k", "model": "m"}],
        "collaboration": {"inflight_timeout": 0.3},
    }))
    pool = WorkerPool(cfg, exclude=None)
    assert pool.names(), "虚拟工人应已进入池中"
    worker = pool.names()[0]

    class HangThenOkClient:
        def __init__(self):
            self.calls = 0

        def chat(self, messages):
            self.calls += 1
            if self.calls == 1:
                time.sleep(2)  # 模拟发起者挂死（远超 inflight_timeout=0.3）
            return {"role": "assistant", "content": "ok"}

    client = HangThenOkClient()
    orig = orch.LLMClient
    orch.LLMClient = lambda *a, **k: client
    try:
        # 发起者：后台线程跑，进入 chat 后挂死 2s
        lead_future: Future = Future()

        def _lead() -> None:
            try:
                lead_future.set_result(pool.ask(worker, "task"))
            except Exception as exc:  # noqa: BLE001
                lead_future.set_exception(exc)

        threading.Thread(target=_lead, name="lead", daemon=True).start()

        # 跟随者：并发发起同 key，应在 ~0.3s 超时后接管并拿结果（而非等满 2s）
        follower_start = time.time()
        follower_result = pool.ask(worker, "task")
        follower_elapsed = time.time() - follower_start

        assert follower_result == "ok"
        assert follower_elapsed < 1.5, f"跟随者不应阻塞到发起者 2s，实际 {follower_elapsed:.2f}s"
        # 接管者重新打网络：发起者(1) + 接管跟随者(1) = 2 次网络调用（不只 1 次、也不无限等待）
        assert client.calls == 2, f"应重新派工一次，实际网络调用 {client.calls} 次"
        # 发起者最终也正常返回（不被接管影响）
        assert lead_future.result(timeout=5) == "ok"
    finally:
        orch.LLMClient = orig


def test_diagnostics_go_to_stderr_not_stdout(capsys):
    """诊断/警告必须走 stderr，stdout 只留给 bridge 与 --stream 的 JSON 行协议。

    实测修正前，配置缺失变量的 `[配置警告]` 直接 print 到 stdout：bridge 启动阶段
    就会往协议流里插入人类可读文本，按行 json.loads 解析的驱动方直接失败。
    """
    from agent.config import _missing_logged

    _missing_logged.clear()
    _interpolate("${MAO_DEFINITELY_MISSING_VAR}")
    captured = capsys.readouterr()
    assert "配置警告" in captured.err, "警告应在 stderr"
    assert captured.out == "", f"stdout 被污染：{captured.out!r}"


def test_session_flush_env_override(monkeypatch):
    # 环境变量应作为 CLI 命令行覆盖的落地机制，优先级高于 config.yaml 的 session 段。
    monkeypatch.setenv("MAO_SESSION_FLUSH_BATCH", "5")
    monkeypatch.setenv("MAO_SESSION_FLUSH_INTERVAL", "0.5")
    # 没有 session 段：完全靠 env 兜底
    cfg = Config(_interpolate({"agents": []}))
    assert cfg.session_flush_batch == 5
    assert cfg.session_flush_interval == 0.5

    # env 覆盖应优先于 config.yaml 的同名字段；未设 env 的字段回落 config
    monkeypatch.setenv("MAO_SESSION_FLUSH_BATCH", "99")
    monkeypatch.delenv("MAO_SESSION_FLUSH_INTERVAL", raising=False)
    cfg2 = Config(_interpolate({"agents": [], "session": {"flush_batch": 16, "flush_interval": 2.0}}))
    assert cfg2.session_flush_batch == 99   # env 赢
    assert cfg2.session_flush_interval == 2.0  # 未设 env 时回落 config


# ---------------- 修正回归：shell=False 引号剥离 / read_file 截断 / _lead 结算兜底 ----------------

def _two_worker_pool() -> WorkerPool:
    cfg = Config(_interpolate({
        "agents": [{"name": "st", "base_url": "http://x", "api_key": "k", "models": ["m1", "m2"]}],
    }))
    return WorkerPool(cfg, exclude=None)


def _install_fake_llm(client_cls):
    """把 orch.LLMClient 换成 fake 工厂，返回还原函数。"""
    orig = orch.LLMClient
    orch.LLMClient = lambda *a, **k: client_cls()
    return orig


def test_try_split_args_strips_wrapping_quotes():
    # shlex posix=False 保留引号字符：不剥离时 `python -c "print(1)"` 会把
    # 带引号的表达式传给 Python（静默无输出）。剥离后应还原真实参数值。
    from agent.tools.builtin import _try_split_args

    args = _try_split_args('python -c "print(1)"')
    assert args == ["python", "-c", "print(1)"]
    # 带空格路径：引号仅作分词，参数值不含引号
    args2 = _try_split_args('type "my file.txt"')
    assert args2 == ["type", "my file.txt"]
    # 无引号命令不受影响
    assert _try_split_args("python -V") == ["python", "-V"]
    # 含 shell 专有语法仍退回 None（走 shell=True）
    assert _try_split_args("dir | findstr x") is None


def test_read_file_truncates_large_file(tmp_path, monkeypatch):
    # read_file 只读头部 64KB：超大文件不得全量载入内存，且返回截断提示。
    import agent.tools.builtin as bi
    from agent.config import load_config

    # 把 ROOT 指到 tmp_path，使测试文件落在"项目根"内（_resolve 越界检查通过）
    monkeypatch.setattr(bi, "ROOT", tmp_path)
    monkeypatch.setattr(bi, "ROOT_RESOLVED", tmp_path.resolve())
    reg = bi.build_builtin_tools(load_config())

    big = tmp_path / "big.txt"
    big.write_text("好" * 100_000, encoding="utf-8")  # 200KB 中文文本

    out = reg.execute("read_file", {"path": "big.txt"})
    assert "已截断" in out
    assert "64KB" in out
    # 输出体量应被限制在 64KB 量级（而非 200KB 全量）
    assert len(out.encode("utf-8")) < bi.MAX_READ_BYTES + 2000


def test_ask_lead_settles_inflight_on_client_build_failure():
    # _lead 结算兜底：LLMClient 构造失败等异常不应让 inflight 悬挂，
    # 跟随者不得等满 inflight_timeout；错误文本返回且 inflight 已清理（可立即重试）。
    cfg = Config(_interpolate({"agents": [{"name": "dummy", "base_url": "http://x", "api_key": "k", "model": "m"}]}))
    pool = WorkerPool(cfg, exclude=None)
    worker = pool.names()[0]

    def _boom(*_a, **_k):
        raise RuntimeError("client build failed")

    orig = orch.LLMClient
    orch.LLMClient = _boom
    try:
        out = pool.ask(worker, "task")
        assert out.startswith("错误：")
        # inflight 已被结算清理：再次调用立即走新一轮（而非跟随者等待）
        assert pool._inflight == {}
        out2 = pool.ask(worker, "task")
        assert out2.startswith("错误：")
    finally:
        orch.LLMClient = orig


# ---------------- ask system 归一（P2-6） ----------------

def test_ask_system_normalization_dedup():
    """system 字段归一后，等价的形式走同一缓存键，避免同语义重复打网络。

    验证两组等价类：
    1) "role-A" 与 "  role-A  "（前后空白） → 同 key
    2) None / "" / "  "（空字符串族）       → 同 key（归一为 None）
    """
    cfg = Config(_interpolate({"agents": [{"name": "dummy", "base_url": "http://x", "api_key": "k", "model": "m"}]}))
    pool = WorkerPool(cfg, exclude=None)
    worker = pool.names()[0]

    calls = {"n": 0}

    class FakeClient:
        def chat(self, messages):
            calls["n"] += 1
            return {"role": "assistant", "content": "ok"}

    orig = orch.LLMClient
    orch.LLMClient = lambda *a, **k: FakeClient()
    try:
        # 第一组：用 "role-A" 入缓存
        r1 = pool.ask(worker, "p", "role-A")
        assert r1 == "ok"
        assert calls["n"] == 1
        # 同 prompt + "role-A" 族（strip 后相同）应命中缓存，不再打网络
        assert pool.ask(worker, "p", "  role-A  ") == "ok"
        assert pool.ask(worker, "p", "role-A") == "ok"
        assert calls["n"] == 1, f"前后空白应命中同一 key，实际 {calls['n']}"

        # 第二组：空串族（None/""/空白）归一后 = None，应走另一条 key
        assert pool.ask(worker, "p", None) == "ok"
        assert calls["n"] == 2
        assert pool.ask(worker, "p", "") == "ok"
        assert pool.ask(worker, "p", "  ") == "ok"
        assert calls["n"] == 2, f"None/空串族应归一为同 key，实际 {calls['n']}"

        # 不同 prompt 走不同的网络
        pool.ask(worker, "p2", "role-A")
        assert calls["n"] == 3
    finally:
        orch.LLMClient = orig


# ---------------- events.to_event error 归一（P2-8） ----------------

def test_to_event_error_normalized():
    """error 类型应显式归一为 {event:error, message:...}，不带 type 残留。"""
    out = to_event({"type": "error", "message": "boom"})
    assert out == {"event": "error", "message": "boom"}
    # message 缺省时也能跑通（不抛异常，输出空串）
    out2 = to_event({"type": "error"})
    assert out2 == {"event": "error", "message": ""}


def test_to_event_unknown_type_passthrough():
    """未识别的 type 仍走兜底透传（兼容自定义事件）。"""
    out = to_event({"type": "custom", "data": "x"})
    assert out == {"event": "custom", "data": "x"}


# ---------------- Agent allow_tools=[] 显式校验（P2-9） ----------------

def test_agent_allow_tools_empty_warns(capsys, tmp_path, monkeypatch):
    """allow_tools=[] 是「禁用全部工具」，应打警告而非静默错用。
    验证构造后 registry 已被清空（任何工具都调不到）。"""
    from agent.core.agent import Agent

    monkeypatch.chdir(tmp_path)  # 隔离 Session 落盘目录
    cfg = Config(_interpolate({
        "agents": [],
        "session": {"flush_batch": 1, "flush_interval": 0.0},
    }))
    bot = Agent(cfg, profile=None, enable_workers=False, allow_tools=[])
    try:
        captured = capsys.readouterr()
        # 警告走 stderr：stdout 要留给 bridge / --stream 的纯 JSON 事件流
        assert "allow_tools=[]" in captured.err, "应在 stderr 打印显式警告"
        assert "allow_tools=[]" not in captured.out, "stdout 不得混入提示"
        # 全部工具都被裁掉
        assert bot.registry.names() == []
        assert bot.removed_tools  # 记录了被裁剪的工具名
    finally:
        bot.close()


def test_agent_allow_tools_none_no_warn(capsys, tmp_path, monkeypatch):
    """allow_tools=None 不应触发警告（放开全部工具）。"""
    from agent.core.agent import Agent

    monkeypatch.chdir(tmp_path)
    cfg = Config(_interpolate({
        "agents": [],
        "session": {"flush_batch": 1, "flush_interval": 0.0},
    }))
    bot = Agent(cfg, profile=None, enable_workers=False, allow_tools=None)
    try:
        captured = capsys.readouterr()
        assert "allow_tools=[]" not in captured.out
        assert "allow_tools=[]" not in captured.err
        # 至少注册了内置工具
        assert "run_shell" in bot.registry.names()
    finally:
        bot.close()


# ---------------- 并行派工确定性 / 整体限时 / 择优（本轮回归） ----------------

def test_collect_and_vote_candidate_order_follows_input():
    """候选编号确定性：collect/vote 的候选顺序必须跟随入参工人顺序，
    不随线程完成先后漂移（第二次调用命中缓存、仅入参顺序不同，最能暴露乱序）。"""
    pool = _two_worker_pool()
    w1, w2 = pool.names()

    class FakeClient:
        def chat(self, messages):
            user = messages[-1]["content"]
            if user.startswith("候选方案"):
                return {"role": "assistant", "content": "1"}
            return {"role": "assistant", "content": "方案A的完整内容"}

    orig = _install_fake_llm(FakeClient)
    try:
        out_ab = pool.vote("题目")                      # 默认池顺序 w1, w2
        out_ba = pool.vote("题目", workers=[w2, w1])    # 同题反序（阶段1命中缓存）
        got_ba = pool.collect([w2, w1], "另一题")
    finally:
        orch.LLMClient = orig

    first_ab = next(ln for ln in out_ab.splitlines() if ln.startswith("[1] "))
    first_ba = next(ln for ln in out_ba.splitlines() if ln.startswith("[1] "))
    assert first_ab.startswith(f"[1] {w1}"), f"默认顺序应 w1 在前：{first_ab}"
    assert first_ba.startswith(f"[1] {w2}"), f"入参反序应 w2 在前：{first_ba}"
    # 共识胜出者随编号确定性变化
    assert f"胜出方案（{w1}）" in out_ab
    assert f"胜出方案（{w2}）" in out_ba
    assert [w for w, _ in got_ba] == [w2, w1]


def test_ask_many_timeout_bounds_total_wait():
    """ask_many 的 timeout 应约束收集阶段总时长：工人挂住时按超时放弃等待
    （原实现 executor 退出 join 全部线程，timeout 实际不约束总时长）。"""
    cfg = Config(_interpolate({"agents": [{"name": "dummy", "base_url": "http://x", "api_key": "k", "model": "m"}]}))
    pool = WorkerPool(cfg, exclude=None)
    worker = pool.names()[0]
    release = threading.Event()

    class HangClient:
        def chat(self, messages):
            release.wait(timeout=5)  # 挂住直到测试放行
            return {"role": "assistant", "content": "ok"}

    orig = _install_fake_llm(HangClient)
    try:
        t0 = time.time()
        out = pool.ask_many([worker], "task", timeout=0.3)
        elapsed = time.time() - t0
        assert elapsed < 2.0, f"ask_many 应被 timeout 限时，实际 {elapsed:.2f}s"
        assert "未完成" in out, "超时工人应以未完成如实带回"
    finally:
        release.set()  # 放行后台线程自行收尾（结算/入缓存不受影响）
        orch.LLMClient = orig


def test_select_best_picks_voted_winner():
    """select_best 复用投票第二步：对已有候选集合投票，返回票数最高者。"""
    pool = _two_worker_pool()

    class FakeClient:
        def chat(self, messages):
            user = messages[-1]["content"]
            if user.startswith("候选方案"):
                return {"role": "assistant", "content": "2"}
            return {"role": "assistant", "content": "unused"}

    orig = _install_fake_llm(FakeClient)
    try:
        picked = pool.select_best([("a", "AAA"), ("b", "BBB")], voters=pool.names())
    finally:
        orch.LLMClient = orig

    assert picked is not None
    winner_w, winner_text, summary = picked
    assert winner_w == "b" and winner_text == "BBB"
    assert "票况" in summary


# ---------------- 命令黑名单：按「命令首词」匹配（本轮修正） ----------------

def test_cmd_blacklist_allows_same_word_in_argument_position():
    """format/clean/del 等词出现在参数位时不得拦截。

    早期实现按"任意位置出现该词"匹配，实测把 `ruff format .`、`make clean`
    这类正常开发命令也拒了，是会影响可用性的真实缺陷。
    """
    from agent.tools.builtin import _check_command_safety

    for cmd in [
        "ruff format .",
        "python -m black --format json x.py",
        "make clean",
        "git log --pretty=format:%H",
        "echo clean",
        "dir | findstr format",
        "python -V",
        "Ruff Format .",
    ]:
        assert _check_command_safety(cmd) is None, f"应放行却被拦：{cmd}"


def test_cmd_blacklist_blocks_dangerous_as_head_word():
    """危险命令落在命令首词位置时仍须拦下（含包装前缀 / 路径 / .exe 后缀归一）。"""
    from agent.tools.builtin import _check_command_safety

    for cmd in [
        "format D:",
        "cmd /c format D:",                    # 剥 cmd /c 包装后判定首词
        "C:/Windows/System32/format.com D:",   # 去路径前缀 + .com 后缀
        "del.exe /s x",
        "del /f /q /s C:/tmp",
        "rd /s /q C:/x",
        "rmdir /s /q C:/x",
        "shutdown /r",
        "diskpart",
        "taskkill /f /im a.exe",
    ]:
        assert _check_command_safety(cmd), f"应拦截却放行：{cmd}"


def test_cmd_blacklist_checks_every_segment_head():
    """单 & / ; 串接时，首词安全但后续段的危险命令也须拦下。

    只查整条命令的首词会漏掉 `echo x & del /s y`（首词 echo，真正删文件的是 del），
    因此按 shell 分隔符切段、逐段查首词。
    """
    from agent.tools.builtin import _check_command_safety

    assert _check_command_safety("echo x & del /s y")
    assert _check_command_safety("echo a ; format D:")
    # 串联的都是安全命令则应放行，不能过度拦截
    assert _check_command_safety("echo a & echo b") is None


def test_cmd_blacklist_encoded_command_abbrev():
    """`-e`/`-enc` 是 -EncodedCommand 的合法缩写，必须拦；
    而 -ExecutionPolicy 不是 encodedcommand 的前缀，不得被误判为编码命令。"""
    from agent.tools.builtin import _check_command_safety

    ps = "power" + "shell "
    for flag in ("-e", "-enc", "-EncodedCommand"):
        msg = _check_command_safety(f"{ps}-NoProfile {flag} ZgBvAHIAbQBhAHQ=")
        assert msg and "EncodedCommand" in msg, f"{flag} 应判为编码命令，实际：{msg}"
    # 精确排除：命中原因是正文里的 format，不是编码命令
    msg2 = _check_command_safety(
        f'{ps}-NoProfile -ExecutionPolicy Bypass -Command "format D:"'
    )
    assert msg2 and "EncodedCommand" not in msg2, f"不应误判为编码命令：{msg2}"
    # 正文安全则放行
    assert _check_command_safety(f'{ps}-NoProfile -Command "Get-Date"') is None


# ---------------- 空 choices 归类为可重试错误（本轮修正） ----------------

def test_llm_empty_choices_classified_retryable():
    """空 choices 必须转成可重试的 LLMError。

    中转站内容过滤会返回 `choices: []`；早期实现直接取 `resp.choices[0]`，
    裸 IndexError 会逃出 LLMClient、绕过上层的错误分类与重试/冷却编排。
    """
    client = LLMClient("http://x", "k", "m", max_retries=0)
    fake = MagicMock()
    fake.chat.completions.create.return_value = SimpleNamespace(choices=[])
    client.client = fake
    with pytest.raises(LLMError) as exc:
        client.chat([{"role": "user", "content": "q"}])
    assert exc.value.error_type == ERR_API
    assert exc.value.retryable is True


def test_llm_empty_choices_retried_then_succeeds(monkeypatch):
    """空 choices 走正常退避重试：下一次拿到正常响应即成功。"""
    client = LLMClient("http://x", "k", "m", max_retries=2)
    monkeypatch.setattr(client, "_backoff", lambda *a, **k: 0)  # 测试不真睡
    state = {"n": 0}

    def create(**_kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return SimpleNamespace(choices=[])  # 模拟内容过滤
        return _fake_response("hi")

    fake = MagicMock()
    fake.chat.completions.create.side_effect = create
    client.client = fake
    assert client.chat([{"role": "user", "content": "q"}])["content"] == "hi"
    assert state["n"] == 2


# ---------------- session_id 路径穿越（本轮修正） ----------------

def test_sanitize_session_id_rejects_unsafe():
    """session_id 会拼成落盘文件名：分隔符 / .. / 盘符 / 超长一律判非法。"""
    from agent.core.session import sanitize_session_id

    assert sanitize_session_id("20260921-120000") == "20260921-120000"
    assert sanitize_session_id("abc_DEF.1-x") == "abc_DEF.1-x"
    for bad in ["", "   ", None, "../../etc/passwd", "a/b", "a\\b", "..", "a..b",
                "C:/x", "x" * 65]:
        assert sanitize_session_id(bad) is None, f"应判非法：{bad!r}"


def test_session_evil_id_stays_inside_sessions_dir(tmp_path, monkeypatch):
    """即使非法 id 透传到 Session，也不得把 JSONL 写到 sessions 目录之外。

    实测修正前 `Session(session_id="../../escaped_sid")` 会把文件写到
    sessions 的同级目录（路径穿越写盘）。
    """
    monkeypatch.setattr(sess_mod, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(sess_mod, "TRASH_DIR", tmp_path / "trash")

    s = Session(session_id="../../escaped", flush_batch=1, flush_interval=0.0)
    s.add("user", "x")
    s.flush()
    assert s._path.parent.resolve() == (tmp_path / "sessions").resolve()
    assert not list(tmp_path.glob("escaped*")), "不得在 sessions 目录外生成文件"


# ---------------- skill 脚本名路径穿越（本轮修正） ----------------

def test_skill_script_path_traversal_rejected(tmp_path):
    """execute_skill_script 的 script 入参必须只是 scripts/ 下的裸 .py 文件名。"""
    from agent.skills_manager import SkillManager

    skill_dir = tmp_path / "demo"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "scripts" / "ok.py").write_text("print('ok')", encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: d\n---\n", encoding="utf-8"
    )
    mgr = SkillManager(skills_dir=tmp_path)
    mgr.scan()
    assert "demo" in mgr.skills

    for bad in ["../../evil.py", "..\\..\\evil.py", "C:/evil.py", "/etc/passwd.py",
                ".hidden.py", "sub/ok.py", "notpy.txt", "ok.py.bak", ""]:
        out = mgr.run_script("demo", bad)
        assert out.startswith("错误：脚本名非法"), f"应拒绝 {bad!r}，实际：{out[:60]}"

    # 合法脚本不受影响，仍能正常执行
    out_ok = mgr.run_script("demo", "ok.py")
    assert "ok" in out_ok and "[exit code] 0" in out_ok


# ---------------- 在飞接管不得清掉新发起者的 inflight（本轮修正） ----------------

def test_ask_takeover_does_not_clear_new_inflight():
    """跟随者超时接管后，旧发起者结算时不得清掉接管者注册的 inflight。

    旧发起者收尾时无条件 `_inflight.pop(key)`，会把接管者的在飞项一并删掉，
    后来者看不到在飞项 → 各自重新打网络，去重在最需要它的挂死场景下正好失效。
    """
    cfg = Config(_interpolate({
        "agents": [{"name": "dummy", "base_url": "http://x", "api_key": "k", "model": "m"}],
        "collaboration": {"inflight_timeout": 0.3},
    }))
    pool = WorkerPool(cfg, exclude=None)
    worker = pool.names()[0]
    key = (worker, "t", None)

    release_lead = threading.Event()
    release_takeover = threading.Event()
    calls = {"n": 0}

    class SlowClient:
        def chat(self, messages):
            calls["n"] += 1
            (release_lead if calls["n"] == 1 else release_takeover).wait(timeout=5)
            return {"role": "assistant", "content": "ok"}

    orig = orch.LLMClient
    orch.LLMClient = lambda *a, **k: SlowClient()
    lead_future: Future = Future()
    follower_future: Future = Future()

    def _run(fut: Future) -> None:
        try:
            fut.set_result(pool.ask(worker, "t"))
        except Exception as exc:  # noqa: BLE001
            fut.set_exception(exc)

    try:
        threading.Thread(target=_run, args=(lead_future,), name="lead", daemon=True).start()
        # 等旧发起者建立 inflight 并进入网络。**必须带 deadline**：任何回归（比如当日
        # 隔离把发起者的 ask 短路、FakeClient 没跑起来）都会让这个 while 空转到天荒地老，
        # 把整套测试挂死、CI 只能靠 timeout 兜底。同文件另外两处并发等待
        # （test_ask_concurrent_dedup_same_key / test_ask_concurrent_initiator_failure_shared_with_followers）
        # 都有 `time.time() < deadline` 保护，这里语义对齐：到期未就位 ⇒ 显式失败。
        lead_deadline = time.time() + 5.0
        while calls["n"] < 1 and time.time() < lead_deadline:
            time.sleep(0.01)
        assert calls["n"] >= 1, "旧发起者未在 5s 内进入网络段，接管路径无法继续（多半是 ask 被上游短路了）"

        threading.Thread(
            target=_run, args=(follower_future,), name="follower", daemon=True
        ).start()
        time.sleep(0.8)  # 等跟随者按 inflight_timeout 接管并注册新 future

        with pool._lock:
            assert len(pool._inflight) == 1, "接管后应有且仅有接管者一条 inflight"
            takeover_fut = pool._inflight[key]

        release_lead.set()  # 放行旧发起者结算
        assert lead_future.result(timeout=5) == "ok"

        with pool._lock:
            remaining = pool._inflight.get(key)
        assert remaining is takeover_fut, "旧发起者不得清掉接管者的 inflight"

        release_takeover.set()  # 放行接管者
        assert follower_future.result(timeout=5) == "ok"
        # 接管者成功入缓存：后续同 key 直接命中，不再打网络
        assert pool.ask(worker, "t") == "ok"
        assert calls["n"] == 2, f"应恰好 2 次网络调用（旧发起者 + 接管者），实际 {calls['n']}"
    finally:
        orch.LLMClient = orig
        release_lead.set()
        release_takeover.set()


# ---------------- 空回复不静默（本轮修正） ----------------

def test_agent_empty_reply_not_silent(tmp_path, monkeypatch):
    """模型返回空 content 且无工具调用时，不得给出一条空白最终答复。"""
    from agent.core.agent import Agent

    monkeypatch.setattr(sess_mod, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(sess_mod, "TRASH_DIR", tmp_path / "trash")
    cfg = Config(_interpolate({
        "agents": [],
        "session": {"flush_batch": 1, "flush_interval": 0.0},
    }))
    bot = Agent(cfg, profile=None, enable_workers=False)
    try:
        bot.llm = SimpleNamespace(
            chat=lambda messages, tools=None: {"role": "assistant", "content": ""}
        )
        out = bot.run("你好")
        assert out.strip(), "不得返回空串"
        assert "空内容" in out
    finally:
        bot.close()


# ---------------- 并行派工实时事件 + 提前采纳（方案A + 结束派工） ----------------

def test_to_event_worker_streaming_schemas():
    """三派工事件在 to_event 里显式归一：preview 截断、ok/计数字段透传。"""
    d = to_event({"type": "worker_dispatch", "workers": ["a", "b"], "prompt": "P" * 600})
    assert d["event"] == "worker_dispatch" and d["total"] == 2
    assert len(d["prompt_preview"]) <= 500
    r = to_event({"type": "worker_result", "worker": "a", "index": 1, "total": 2,
                  "elapsed_ms": 1234, "ok": True, "answer": "hi"})
    assert r["event"] == "worker_result" and r["worker"] == "a"
    assert r["ok"] is True and r["preview"] == "hi" and r["elapsed_ms"] == 1234
    r2 = to_event({"type": "worker_result", "worker": "b", "ok": False, "answer": "错误：隔离"})
    assert r2["ok"] is False
    c = to_event({"type": "worker_gather_cancelled", "completed": ["a"], "skipped": ["b"]})
    assert c["event"] == "worker_gather_cancelled"
    assert c["completed"] == ["a"] and c["skipped"] == ["b"]


def test_functiontool_injects_ctx_only_for_two_arg_funcs():
    """FunctionTool 按签名决定是否注入 ctx：双参收、单参不收；单参工具无 ctx 也照常跑。"""
    seen = {}

    def needs_ctx(a, ctx=None):
        seen["ctx"] = ctx
        return "yes"

    def simple(a):
        return "no"

    reg = ToolRegistry()
    reg.register(FunctionTool("n", "d", {}, needs_ctx))
    reg.register(FunctionTool("s", "d", {}, simple))
    sentinel = SimpleNamespace(emit=lambda e: None, cancel_event=None)
    assert reg.execute("n", {}, sentinel) == "yes"
    assert seen["ctx"] is sentinel, "双参工具必须收到 ctx"
    assert reg.execute("s", {}) == "no"          # 无 ctx 路径（bridge call_tool/单测）
    assert reg.execute("s", {}, sentinel) == "no"  # 传了 ctx 也不报错、忽略之


def _two_worker_stream_pool() -> WorkerPool:
    cfg = Config(_interpolate({"agents": [
        {"name": "fast", "base_url": "u", "api_key": "k", "model": "fast"},
        {"name": "slow", "base_url": "u", "api_key": "k", "model": "slow"},
    ]}))
    return WorkerPool(cfg, exclude=None)


def test_ask_many_emits_worker_events(monkeypatch):
    """ask_many 带 emit：先 worker_dispatch，再每个工人各一条 worker_result（含 ok/answer）。"""
    class Instant:
        def __init__(self, *_a, **_k):
            pass

        def chat(self, messages, tools=None):
            return {"content": "ok"}

    monkeypatch.setattr(orch, "LLMClient", Instant)
    pool = _two_worker_stream_pool()
    evs: list[dict] = []
    out = pool.ask_many(["fast", "slow"], "任务", emit=lambda e: evs.append(e))
    types = [e["type"] for e in evs]
    assert types[0] == "worker_dispatch"
    assert types.count("worker_result") == 2
    wr = [e for e in evs if e["type"] == "worker_result"]
    assert all(e["ok"] and e["answer"] == "ok" for e in wr)
    assert {e["worker"] for e in wr} == {"fast", "slow"}
    assert "### 工人 fast 的结果" in out and "### 工人 slow 的结果" in out
    # 无 emit 时（bridge/单测）退化成纯栅栏调用，不抛异常
    assert "工人 fast 的结果" in pool.ask_many(["fast", "slow"], "任务2")


def test_ask_many_cancel_returns_completed_early(monkeypatch):
    """快工人一回来就置取消位 → 慢工人被放弃等待、结果里标注提前结束（确定性、无计时竞态）。"""
    class Timed:
        def __init__(self, _b, _a, model, **_k):
            self.model = model

        def chat(self, messages, tools=None):
            if self.model == "slow":
                time.sleep(2.0)  # 慢节点：取消后应放弃等待它（此线程后台自行收尾）
            return {"content": f"ans-{self.model}"}

    monkeypatch.setattr(orch, "LLMClient", Timed)
    pool = _two_worker_stream_pool()
    cancel = threading.Event()
    evs: list[dict] = []

    def emit(e: dict) -> None:
        evs.append(e)
        if e["type"] == "worker_result" and e["worker"] == "fast":
            cancel.set()  # 采纳已完成的 fast，结束对 slow 的等待

    t0 = time.monotonic()
    out = pool.ask_many(["fast", "slow"], "任务", emit=emit, cancel_event=cancel)
    dt = time.monotonic() - t0
    assert dt < 1.5, f"取消未生效：等了 {dt:.2f}s（慢节点未收尾就该放弃等待）"
    assert "ans-fast" in out, "已完成工人的结果必须保留"
    assert "提前结束" in out and "slow" in out
    assert any(e["type"] == "worker_gather_cancelled" for e in evs)
