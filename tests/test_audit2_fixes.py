"""第二轮全链路审计（2026-09-22）八项修复的回归测试：全部离线。

对应问题（编号见 docs/2026-09-22-工作记录.md 第三轮）：
1. 投票共识比例的分母混入无效票（投越界编号）
2. 健康档案在锁内做文件 IO，经 pool 锁串联阻塞整条派工路径
3. 数值配置项裸 int()/float()，类型写错直接崩启动
4. 流水线阶段之间不做长度约束，长草稿把下一阶段 prompt 顶出模型窗口
5. write_file 静默覆盖不可逆
6. CLI /new 把用户选的主智能体换成池内第一个
7. 注入模式误拦引号内的字面量（echo "a|b|c"）
8. lifespan 收尾漏清 _session_orch
另附：408/425 归入可重试（免费中转网关超时常见）。

用例均针对缺陷构造（去掉修复即失败），配套 tests/mutation_check.py 的变异校验。
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import openai
import pytest

import agent.core.orchestrator as orch
import agent.tools.builtin as builtin
from agent.config import Config, _interpolate
from agent.core.health import get_health
from agent.core.llm import LLMClient
from agent.core.orchestrator import WorkerPool


def _pool(models: str, cap: int = 5, **collab) -> WorkerPool:
    cfg = Config(_interpolate({
        "agents": [{"name": "st", "base_url": "u", "api_key": "k", "models": models}],
        "collaboration": {"max_participants": cap, **collab},
    }))
    return WorkerPool(cfg, exclude=None)


# ---------------- 1) 投票分母只算有效票 ----------------

def test_ballot_denominator_excludes_invalid_votes(monkeypatch):
    """3 张票里 2 张投不存在的编号：有效票 1/1 就该达成共识，不该被判 1/3。"""
    pool = _pool("m1,m2,m3")
    answers = ["9", "9", "1"]  # 9 号不存在（候选只有 3 份）
    lock = threading.Lock()
    it = iter(answers)

    class FakeClient:
        def chat(self, messages):
            if messages[-1]["content"].startswith("候选方案"):
                with lock:
                    v = next(it, "1")
                return {"role": "assistant", "content": v}
            return {"role": "assistant", "content": "候选方案正文"}

    monkeypatch.setattr(orch, "LLMClient", lambda *a, **k: FakeClient())
    v = pool.vote("选一个方案", workers=pool.names(), threshold=0.5)
    assert v["ok"] is True and v["consensus"] is True, "有效票 1/1 过半 ⇒ ok+consensus 均 True"
    out = v["report"]
    assert "共识达成" in out, f"有效票 1/1 应达成共识：\n{out}"
    assert "1/1" in out
    assert "投了不存在的编号" in out, "废票要如实提示，不能静默丢弃"


def test_ballot_all_invalid_returns_none(monkeypatch):
    """全部投废票 ⇒ 没有有效票 ⇒ 返回 None，不得谎报共识。"""
    pool = _pool("m1,m2")

    class FakeClient:
        def chat(self, messages):
            return {"role": "assistant", "content": "99 号"}

    monkeypatch.setattr(orch, "LLMClient", lambda *a, **k: FakeClient())
    assert pool.select_best([("a", "AAA"), ("b", "BBB")], voters=pool.names()) is None


# ---------------- 2) 健康档案的文件 IO 不在锁内 ----------------

def test_persist_does_not_block_quarantine_queries(tmp_path, monkeypatch):
    """写盘进行中，quarantined（派工热路径）必须能立即返回。

    这里刻意让**真实写盘**变慢（换掉 os.replace），而不是替换 _write_payload ——
    替换掉整个写盘方法的话，"IO 是否在锁内"这个被测点就被绕过去了（变异测试
    对此报过恒真）。
    """
    import agent.core.health as health_mod

    h = get_health(tmp_path / "h.json")
    started = threading.Event()
    real_replace = health_mod.os.replace

    def slow_replace(src, dst):
        started.set()
        time.sleep(0.4)  # 模拟慢盘
        return real_replace(src, dst)

    monkeypatch.setattr(health_mod.os, "replace", slow_replace)
    worker = threading.Thread(target=lambda: h.record_failure("w", "m", ""))
    worker.start()
    assert started.wait(timeout=3), "写盘应被调用"

    t0 = time.perf_counter()
    h.quarantined("other")
    waited_ms = (time.perf_counter() - t0) * 1000
    worker.join(timeout=5)

    assert waited_ms < 200, f"写盘进行中查询被阻塞 {waited_ms:.1f}ms —— IO 又回到锁内了"


def test_concurrent_record_failure_keeps_store_parsable(tmp_path):
    """写盘移到锁外后并发落盘不得互相踩：档案要能解析、不留临时文件。"""
    import json

    store = tmp_path / "h.json"
    h = get_health(store)

    def writer(idx: int) -> None:
        for j in range(20):
            h.record_failure(f"mt:{idx}-{j}", "m1", "")

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data = json.loads(store.read_text(encoding="utf-8"))
    assert len(data["models"]) == 100
    assert not list(tmp_path.glob("*.tmp")), "失败路径要清掉临时文件"


# ---------------- 3) 数值配置类型容错 ----------------

@pytest.mark.parametrize(("attr", "data", "expected"), [
    ("max_participants", {"collaboration": {"max_participants": "five"}}, 5),
    ("max_workers", {"collaboration": {"max_workers": "many"}}, 3),
    ("shell_timeout", {"tools": {"shell_timeout": "soon"}}, 120),
])
def test_bad_numeric_config_falls_back(attr, data, expected, capsys):
    """类型写错不得抛异常：回退默认值并在 stderr 给一次明确告警。"""
    assert getattr(Config(data), attr) == expected, f"{attr} 应回退默认值"
    assert "配置警告" in capsys.readouterr().err


def test_valid_numeric_config_untouched():
    cfg = Config({"collaboration": {"max_participants": 9}, "tools": {"shell_timeout": 30}})
    assert cfg.max_participants == 9
    assert cfg.shell_timeout == 30


# ---------------- 4) write_file 覆盖前留档 ----------------

def test_write_file_backs_up_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(builtin, "ROOT", tmp_path)
    monkeypatch.setattr(builtin, "ROOT_RESOLVED", tmp_path.resolve())
    monkeypatch.setattr(builtin, "BACKUP_DIR", tmp_path / "backup")
    registry = builtin.build_builtin_tools(Config({}))

    target = tmp_path / "important.txt"
    target.write_text("原始重要内容", encoding="utf-8")
    out = registry.execute("write_file", {"path": "important.txt", "content": "覆盖后的内容"})

    assert target.read_text(encoding="utf-8") == "覆盖后的内容"
    assert "留档" in out, f"返回里要如实告知留档位置：{out}"
    backups = list((tmp_path / "backup").glob("*.bak"))
    assert len(backups) == 1, f"应留一份档：{backups}"
    assert backups[0].read_text(encoding="utf-8") == "原始重要内容", "留档必须是覆盖前的内容"


def test_write_file_new_file_needs_no_backup(tmp_path, monkeypatch):
    monkeypatch.setattr(builtin, "ROOT", tmp_path)
    monkeypatch.setattr(builtin, "ROOT_RESOLVED", tmp_path.resolve())
    monkeypatch.setattr(builtin, "BACKUP_DIR", tmp_path / "backup")
    registry = builtin.build_builtin_tools(Config({}))
    out = registry.execute("write_file", {"path": "brand_new.txt", "content": "x"})
    assert "留档" not in out, "新建文件没有可留档的旧内容"
    assert not (tmp_path / "backup").exists()


# ---------------- 6) 注入检查不再误拦引号内字面量 ----------------

def test_injection_check_ignores_quoted_literals():
    from agent.tools.builtin import _check_command_safety

    assert _check_command_safety('echo "a|b|c"') is None, "引号内的管道是字面量"
    assert _check_command_safety('python -c "print(1|2)"') is None
    assert _check_command_safety("findstr \"a|b|c\" x.txt") is None


def test_injection_check_still_blocks_real_concatenation():
    from agent.tools.builtin import _check_command_safety

    assert _check_command_safety("echo x && del /s y") is not None
    assert _check_command_safety("a | b | c") is not None
    assert _check_command_safety('echo "x" && del y') is not None, "引号外的拼接照拦"


# ---------------- 7) 408/425 可重试 ----------------

@pytest.mark.parametrize(("code", "expect_retry"), [(408, True), (425, True), (429, True),
                                                    (503, True), (404, False), (401, False)])
def test_http_status_retry_classification(code, expect_retry, monkeypatch):
    calls = {"n": 0}

    def make_err(_code: int):
        resp = MagicMock()
        resp.status_code = _code
        return openai.APIStatusError(f"http {_code}", response=resp, body=None)

    def create(_code=code, **_kwargs):
        calls["n"] += 1
        raise make_err(_code)

    monkeypatch.setattr(LLMClient, "_backoff", staticmethod(lambda attempt, error_type: 0.0))
    client = LLMClient("http://x", "k", "m", max_retries=1)
    monkeypatch.setattr(client, "client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ))

    with pytest.raises(Exception):  # noqa: B017 - 断言的是重试次数而非异常类型
        client.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == (2 if expect_retry else 1), f"HTTP {code} 的重试次数不符预期"
