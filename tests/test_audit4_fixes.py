"""第四轮全库审计（2026-10-09）修复的回归测试：全部离线。

对应问题（按严重度）：
- P0：`bash -c "rm -rf ~"` 绕过三层命令防护（解释器内联集合缺 POSIX shell 家族）→
      见 test_core.py::test_posix_shell_inline_guard_*
- P0：单个 SKILL.md 读失败炸掉整个技能扫描（进而炸启动）
- P1：强制探测 `probe_all` 绕过 `_claim` 名额守卫，两轮同时在打网络（污染延迟档案）
- P1：客户端缓存与池记账共用一把锁（建 httpx/SSL 期间把纯内存记账串行化）
- P1：兜底异常原文外流到 stdout 协议行与工人上下文（凭证形态的串没抹）
- P2：进程内时限混用墙上时钟（NTP 校时即可提前解除冷却）
- P2：三处逐字重复的"锁内快照→tmp→os.replace"与两份共享注册表（修一处漏两处）
- P2：`.env` 引号内注释被当注释剥离；dotenv 双层重复解析
- P2：投票阈值/投票人名单/站并行度/下线判定各自两份口径

用例都针对缺陷构造（去掉修复即失败），配套 tests/mutation_check.py 的变异校验。
"""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest

import agent.core.orchestrator as orch
import agent.tools.builtin as builtin
from agent.config import Config, _interpolate, _load_dotenv
from agent.core.health import ModelHealth
from agent.core.latency import LatencyStore
from agent.core.llm import LLMClient, LLMError
from agent.core.orchestrator import WorkerPool
from agent.core.redact import safe_error_text
from agent.model_registry import RETIRE_ALREADY, RETIRE_MISSING, RETIRE_OFFLINED, retire_model


def _pool(models: str, cap: int = 5, **collab) -> WorkerPool:
    cfg = Config(
        _interpolate(
            {
                "agents": [{"name": "st", "base_url": "u", "api_key": "k", "models": models}],
                "collaboration": {"max_participants": cap, **collab},
            }
        )
    )
    return WorkerPool(cfg, exclude=None)


def _two_station_pool(**collab) -> WorkerPool:
    """两个站各挂两个模型：同站串行下"跨站才真并行"的用例需要真有两个站。"""
    cfg = Config(
        _interpolate(
            {
                "agents": [
                    {
                        "name": "sa",
                        "base_url": "http://a/v1",
                        "api_key": "ka",
                        "models": "m-a1,m-a2",
                    },
                    {
                        "name": "sb",
                        "base_url": "http://b/v1",
                        "api_key": "kb",
                        "models": "m-b1,m-b2",
                    },
                ],
                "collaboration": {"max_workers": 3, "max_participants": 4, **collab},
            }
        )
    )
    return WorkerPool(cfg, exclude=None)


# ---------------- 1) 落盘骨架：取快照必须在 _write_lock 之内 ----------------


@pytest.mark.parametrize("cls", [ModelHealth, LatencyStore], ids=["health", "latency"])
def test_snapshot_is_taken_inside_the_write_lock(tmp_path, monkeypatch, cls):
    """写盘顺序必须等于快照顺序（每次落盘都是上一次的超集），靠"持锁取快照"保证。

    这不是靠多线程抢时序来验证的（快机器上并发写常常碰不到，等于没测）：直接试探
    `_write_lock` 能否被**自己**再次拿到。普通 Lock 不可重入 ⇒ 拿不到就说明此刻正持着它，
    即快照与落盘在同一段串行区里；一旦有人把取快照挪到锁外（历史上的 lost update 写法），
    这里立刻可拿 ⇒ 断言失败。
    """
    store = cls(tmp_path / "ledger.json")
    if cls is ModelHealth:
        store.record_failure("alpha", model="m-a", models_env="A_MODELS", day="2026-01-01")
    else:
        store.record("alpha", 12.0)

    held: list[bool] = []
    real = store._snapshot_locked

    def probe():
        text = real()
        got = store._write_lock.acquire(blocking=False)
        if got:
            store._write_lock.release()
        held.append(not got)
        return text

    monkeypatch.setattr(store, "_snapshot_locked", probe)
    store.write()
    assert held == [True], "取快照时并未持有 _write_lock —— 旧快照可能覆盖新快照丢记账"


def test_ledger_write_keeps_file_parsable_and_no_tmp_left(tmp_path):
    """落盘失败/并发都不留 .tmp 垃圾，档案始终是合法 JSON（原子替换语义）。"""
    store = LatencyStore(tmp_path / "lat.json")
    for i in range(30):
        store.record(f"w{i}", 10.0 + i)
    files = list(tmp_path.iterdir())
    assert files == [tmp_path / "lat.json"], f"不应留下临时文件：{files}"
    raw = json.loads((tmp_path / "lat.json").read_text(encoding="utf-8"))
    assert set(raw["models"]) >= {"w0", "w29"}


# ---------------- 2) 错误文案脱敏 ----------------


def test_redact_masks_credential_shapes():
    """四种凭证形态都要抹掉，且空值/超长有兜底（这段文本会进 stdout 协议行）。"""
    out = safe_error_text(
        "failed url=https://gw.test/v1/chat?api_key=sk-abcdefgh123456&x=1 "
        "header: Bearerghijklmnop0123 key: TOPSECRET99"
    )
    assert "abcdefgh123456" not in out, f"URL query 里的 key 泄漏：{out}"
    assert "x=1" not in out, "query 段应整段隐去"
    assert "TOPSECRET99" not in out, f"key: 形式泄漏：{out}"
    assert "已隐去" in out
    assert safe_error_text("") == "未知错误"
    assert safe_error_text(None) == "未知错误"
    long_text = safe_error_text("x" * 500, limit=50)
    assert long_text.endswith("…（已截断）") and len(long_text) < 60


def test_redact_keeps_plain_message_readable():
    """脱敏不能把有用的信息抹成人话不通：普通错误文本要原样保留。"""
    assert safe_error_text("服务端错误 HTTP 503") == "服务端错误 HTTP 503"


def test_llm_fallback_error_does_not_leak_key(monkeypatch):
    """底层异常常回显请求 URL：兜底分支透传原文前必须过脱敏（否则泄进主与工人上下文）。"""
    secret = "sk-REALSECRETVALUE123456"

    class Boom:
        def __init__(self, *a, **k):
            pass

        def chat_once(self, messages):
            raise RuntimeError(f"400 Bad Request: https://gw.test/v1?api_key={secret}")

    pool = _pool("m1")
    monkeypatch.setattr(orch, "LLMClient", lambda *a, **k: Boom())
    res = pool.ask_result(pool.names()[0], "hi")
    assert res["ok"] is False
    assert secret not in res["error"], f"派工错误文案泄漏凭证：{res['error']}"


def test_llm_client_wraps_unknown_error_redacted(monkeypatch):
    """LLMClient.chat 的兜底分支同样先抹再抛（ERR_OTHER 的文本会被上层原样带出）。"""
    secret = "sk-REALSECRETVALUE123456"
    client = LLMClient("http://x/v1", "k", "m", timeout=5, max_retries=0)

    def boom(**kwargs):
        raise RuntimeError(f"bad gateway: https://gw.test/v1?api_key={secret}")

    monkeypatch.setattr(client.client.chat.completions, "create", boom)
    with pytest.raises(LLMError) as exc:
        client.chat([{"role": "user", "content": "hi"}])
    assert secret not in str(exc.value), f"LLMError 文本泄漏凭证：{exc.value}"


# ---------------- 3) 冷却时限不跟墙上时钟走 ----------------


def test_cooldown_is_immune_to_wall_clock_jump(monkeypatch):
    """NTP 校时/时区切换会让 time.time() 一步跳很大；进程内时限必须用 monotonic。

    旧写法用 time.time() 比较：时钟向前跳 ⇒ 冷却被提前解除、坏节点立刻又被派工；
    向后跳 ⇒ 冷却迟迟不到期，好节点白白被跳过。这里把墙上时钟拨到一年后做确定判定。
    只替换 orchestrator 模块看到的 time（不动全局 stdlib，避免影响 pytest 自己的计时）。
    """
    pool = _pool("m1", cooldown_fails=1, cooldown_base=3600)
    worker = pool.names()[0]
    pool._record_result(worker, ok=False)
    assert pool._skip_reason(worker) == "冷却中"

    real_time = time.time
    monkeypatch.setattr(
        orch,
        "time",
        SimpleNamespace(
            monotonic=time.monotonic,
            time=lambda: real_time() + 365 * 24 * 3600,
        ),
    )
    assert pool._skip_reason(worker) == "冷却中", (
        "墙上时钟一跳就解除冷却 —— 时限又用回 time.time() 了"
    )
    snap = pool.health_status()
    assert snap[0]["available"] is False, "health 快照同样不得被时钟跳变骗过"
    assert snap[0]["cooldown_s"] > 3000, f"剩余冷却时长应仍接近 3600s：{snap[0]}"


# ---------------- 4) 客户端缓存锁与池记账锁分离 ----------------


def test_client_construction_does_not_hold_pool_lock(monkeypatch):
    """建 LLMClient 要建 httpx 客户端与 SSL 上下文（毫秒~几十毫秒），不能压着记账锁。

    同一个试探手法：构造期间池记账锁 self._lock 必须**可以**被拿到（记账不被串行化），
    而缓存锁 _clients_lock 必须**不能**被拿到（首次并发创建仍互斥，不会各建一套连接池）。
    """
    pool = _pool("m1,m2")
    seen: dict[str, bool] = {}

    class ProbeClient:
        def __init__(self, *a, **k):
            free_accounting = pool._lock.acquire(blocking=False)
            cached_locked = pool._clients_lock.acquire(blocking=False)
            if free_accounting:
                pool._lock.release()
            if cached_locked:
                pool._clients_lock.release()
            seen["accounting_free"] = free_accounting
            seen["clients_held"] = not cached_locked

    monkeypatch.setattr(orch, "LLMClient", ProbeClient)
    profile = pool.profiles[pool.names()[0]]
    pool._client_for(profile)
    assert seen.get("accounting_free") is True, "建客户端期间持着池记账锁，把纯内存记账串行化了"
    assert seen.get("clients_held") is True, "建客户端时没持缓存锁，并发首派会各建一套连接池"


# ---------------- 5) 同站串行门 ----------------


def test_station_parallel_cap_is_one_formula():
    """并行度口径只有一份：min(max_workers, max_per_station × 站数)。

    历史上批次数量与 concurrency_status 各算各的（一处折算站数、一处不折），主看到的
    "能并行几路"与实际派工批次不一致 → 排队时间被整体限时误判成"尾部批次未完成"。
    """
    pool = _two_station_pool()
    assert pool._station_parallel_cap(1) == 1
    assert pool._station_parallel_cap(2) == 2
    assert pool._station_parallel_cap(9) == 3, "受 max_workers=3 封顶"

    zero_cap = _two_station_pool()
    zero_cap._max_per_station = 0  # 配置写 0/负数也要退化成每站 1 路
    assert zero_cap._station_parallel_cap(2) == 2


def test_concurrency_status_matches_batch_concurrency():
    """主预检拿到的口径必须等于派工实际用的批次数口径（同一公式，两份答案不行）。"""
    pool = _two_station_pool()
    status = pool.concurrency_status()
    names = pool.names()
    assert status["stations"] == 2
    assert status["effective"] == pool._batch_concurrency(len(names), names) == 2


def test_chat_gated_releases_station_gate_on_failure():
    """请求失败也要放门：门卡在发请求那一层，漏一次释放就整站永久堵死。

    回退递归必须发生在门外（持 A 站门等 B 站门会互相等成死锁），所以这里额外确认
    _chat_gated 只包**一次** chat、返回时门已释放。
    """
    pool = _pool("m1")
    station = ("http://a/v1", "ka")
    gate = pool._station_gate(station)

    class Boom:
        def chat_once(self, messages):
            raise RuntimeError("boom")

    resp, err, ms = pool._chat_gated(Boom(), station, [{"role": "user", "content": "x"}])
    assert resp is None and isinstance(err, RuntimeError) and ms >= 0
    assert gate.acquire(blocking=False), "请求失败后站门没释放"
    gate.release()


def test_probe_yields_when_station_gate_is_busy(monkeypatch):
    """探测让路：站门被真实请求占着时本轮跳过该工人，绝不陪等几分钟、也不污染样本。"""
    import agent.core.latency as lat_mod

    monkeypatch.setattr(lat_mod, "_PROBE_GATE_WAIT", 0.05)  # 免让路等待真占 2s
    pool = _pool("m1")
    worker = pool.names()[0]
    profile = pool.profiles[worker]

    def never(path):  # 一旦真去取探测客户端就是没让路
        raise AssertionError("站门被占着时不该发探测请求")

    monkeypatch.setattr(pool, "_probe_client_for", never)
    gate = pool._station_gate((profile.base_url, profile.api_key))
    assert gate.acquire(blocking=False)  # 模拟该站正在跑真实派工
    try:
        assert pool._probe_one(worker) is None
    finally:
        gate.release()


# ---------------- 6) .env 解析：引号优先于注释剥离 ----------------


def test_dotenv_keeps_hash_inside_quotes(tmp_path, monkeypatch):
    """`KEY="a #b"` 的 # 属于值本身；只有未加引号时才剥"空格+#"起的注释。

    旧实现先 split(" #") 再 strip 引号，把带 # 的值静默截断（如端点带 query 片段、
    模型清单带屏蔽标记），用户看到的是"改了没生效"。
    """
    keys = ["AUDIT4_Q1", "AUDIT4_Q2", "AUDIT4_PLAIN", "AUDIT4_MARK", "AUDIT4_UNCLOSED"]
    for k in keys:
        monkeypatch.delenv(k, raising=False)
    env_file = tmp_path / "dotenv"
    env_file.write_text(
        'AUDIT4_Q1="a #b"\n'
        "AUDIT4_Q2='x #y'\n"
        "AUDIT4_PLAIN=120 # 秒\n"
        "AUDIT4_MARK=#m-one\n"
        'AUDIT4_UNCLOSED="c #d\n',
        encoding="utf-8",
    )
    try:
        names = _load_dotenv(env_file, set())
        assert sorted(names) == sorted(keys)
        assert os.environ["AUDIT4_Q1"] == "a #b", "引号内的 # 被当成注释剥掉了"
        assert os.environ["AUDIT4_Q2"] == "x #y"
        assert os.environ["AUDIT4_PLAIN"] == "120", "未加引号时行内注释仍要剥掉"
        assert os.environ["AUDIT4_MARK"] == "#m-one", "紧贴值的 # 是下线标记，不是注释"
        assert os.environ["AUDIT4_UNCLOSED"] == "c #d", "未闭合引号取起始引号之后的全部"
    finally:
        for k in keys:
            os.environ.pop(k, None)


# ---------------- 7) 工具层：目录不是文件 ----------------


def test_write_and_append_reject_directory(tmp_path, monkeypatch):
    """写目录要回到人话错误，而不是 IsADirectoryError 的技术栈（read_file 已有同款）。"""
    monkeypatch.setattr(builtin, "ROOT", tmp_path)
    monkeypatch.setattr(builtin, "ROOT_RESOLVED", tmp_path.resolve())
    monkeypatch.setattr(builtin, "BACKUP_DIR", tmp_path / "backup")
    registry = builtin.build_builtin_tools(Config({}))
    (tmp_path / "adir").mkdir()

    out = registry.execute("write_file", {"path": "adir", "content": "x"})
    assert "是目录" in str(out), f"write_file 对目录应给可读错误：{out}"
    assert "Error" not in str(out) and "Traceback" not in str(out)
    out2 = registry.execute("append_file", {"path": "adir", "content": "x"})
    assert "是目录" in str(out2), f"append_file 对目录应给可读错误：{out2}"


# ---------------- 8) 下线判定只有一份口径 ----------------


def test_retire_model_reports_three_states(tmp_path):
    """下线改写返回三态（新下线 / 早已带 # / 清单里没有），调用方据此给不同提示。

    旧实现是 bool + health 侧再自己解析一遍文件判断"是否已下线"（两份口径、两次读盘，
    还出现"已下线的模型再失败时提示未找到条目"的错话）。
    """
    reg = tmp_path / "model_registry.txt"
    reg.write_text("NODE_A_MODELS=m-one,#m-two@256k\n", encoding="utf-8")

    assert retire_model(reg, "NODE_A_MODELS", "m-one") == RETIRE_OFFLINED
    assert "#m-one" in reg.read_text(encoding="utf-8")
    assert retire_model(reg, "NODE_A_MODELS", "m-two") == RETIRE_ALREADY, "已带 # 视为已处理"
    assert retire_model(reg, "NODE_A_MODELS", "m-three") == RETIRE_MISSING
    assert retire_model(reg, "NODE_X_MODELS", "m-one") == RETIRE_MISSING
    assert retire_model(tmp_path / "nope.txt", "NODE_A_MODELS", "m-one") == RETIRE_MISSING


def test_health_retire_notice_says_offlined_when_already_marked(tmp_path):
    """连击达阈值 + 模型早已带 # ⇒ 提示"已下线"，不能说"未找到对应条目"。"""
    reg = tmp_path / "model_registry.txt"
    reg.write_text("A_MODELS=#m-a\n", encoding="utf-8")
    h = ModelHealth(tmp_path / "h.json", registry_path=reg, retire_days=2)
    for day in ("2026-01-01", "2026-01-02"):
        h.note_active(day)
        notice = h.record_failure("w-a", model="m-a", models_env="A_MODELS", day=day)
    assert notice and "标 # 下线" in notice, f"应给已下线提示：{notice}"
    assert "未找到" not in notice
