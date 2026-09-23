"""方案 A 回归测试：主智能体外部算好的方案（master_contribution）参与投票/择优。

背景：已确认"外部 AI 当主"（bridge 模式），MAO 侧无法反向调用主，所以"主也参与
干活"落地为——主把方案**文本**传进来，作为"只入候选、不出力"的候选追加到投票/
择优的候选池末尾（编号=工人候选数+1，不影响工人确定性编号）。本文件覆盖：
- 主方案进候选且编号靠后、工人编号不变（确定性）
- 主方案可被工人投中、作为胜出方案回吐
- 空值/非字符串视为未提供（行为与不传一致，向后兼容）
- 工人全失败但有主方案 → 主方案兜底直接胜出（不再报"无可用方案"）
- bridge ask_vote 透传 master_contribution
"""
from __future__ import annotations

import agent.core.orchestrator as orch
from agent.config import Config, _interpolate
from agent.core.orchestrator import WorkerPool


def _two_worker_pool() -> WorkerPool:
    """一站两模型的最小池（与 test_core 的同名设施一致的轻量构造）。"""
    cfg = Config(_interpolate({
        "agents": [{"name": "st", "base_url": "http://x", "api_key": "k", "models": ["m1", "m2"]}],
    }))
    return WorkerPool(cfg, exclude=None)


def _install_fake_llm(client_cls):
    """把 orch.LLMClient 换成 fake 工厂，返回还原函数（同 test_core）。"""
    orig = orch.LLMClient
    orch.LLMClient = lambda *a, **k: client_cls()
    return orig


def test_master_contribution_appended_and_worker_order_stable():
    """主方案进候选：追加到末尾（编号 > 工人数），工人候选编号与顺序不受影响。"""
    pool = _two_worker_pool()
    w1, w2 = pool.names()

    class FakeClient:
        def chat(self, messages):
            user = messages[-1]["content"]
            if user.startswith("候选方案"):
                return {"role": "assistant", "content": "1"}
            return {"role": "assistant", "content": "工人方案"}

    orig = _install_fake_llm(FakeClient)
    try:
        out = pool.vote("题目", master_contribution="主智能体的完整方案")
    finally:
        orch.LLMClient = orig

    report = out["report"]
    # 工人候选编号 1..N，主方案为 N+1（@master 标记、在末尾）
    assert f"[1] {w1}" in report
    assert f"[2] {w2}" in report
    assert "[3] @master" in report
    # 候选人全部参与计票；胜出追踪到主方案出现在报告里
    assert "@master" in report


def test_master_contribution_can_win_ballot():
    """主方案可被选为胜出方：工人投票倾向主方案编号（末尾）时回吐其全文。"""
    pool = _two_worker_pool()

    class FakeClient:
        def chat(self, messages):
            user = messages[-1]["content"]
            if user.startswith("候选方案"):
                return {"role": "assistant", "content": "3"}  # 全部投主方案
            return {"role": "assistant", "content": "工人方案"}

    orig = _install_fake_llm(FakeClient)
    try:
        out = pool.vote("题目", master_contribution="主方案全文AAA")
    finally:
        orch.LLMClient = orig

    assert out["ok"] and out["consensus"]
    assert "胜出方案（@master）" in out["report"]
    assert "主方案全文AAA" in out["report"]


def test_master_contribution_empty_and_falsy_ignored():
    """空/None/非字符串串按未提供处理：报告与不带主方案逐字一致。"""
    pool = _two_worker_pool()
    w1 = pool.names()[0]

    class FakeClient:
        def chat(self, messages):
            user = messages[-1]["content"]
            if user.startswith("候选方案"):
                return {"role": "assistant", "content": "1"}
            return {"role": "assistant", "content": "工人方案"}

    orig = _install_fake_llm(FakeClient)
    try:
        none_baseline = pool.vote("题目")
        empty = pool.vote("题目", master_contribution="")
        blank = pool.vote("题目", master_contribution="   ")
        nonstr = pool.vote("题目", master_contribution=123)  # 非字符串：按 str() 后空白 → 忽略
    finally:
        orch.LLMClient = orig

    assert empty["report"] == none_baseline["report"]
    assert blank["report"] == none_baseline["report"]
    assert nonstr["report"] == none_baseline["report"]
    assert "[1] @master" not in none_baseline["report"]
    assert f"[1] {w1}" in none_baseline["report"]


def test_master_contribution_backstops_when_all_workers_fail():
    """全败兜底：工人一个候选都没有、只有主方案时直接胜出，不报"无可用方案"。"""
    pool = _two_worker_pool()

    class BoomClient:
        def __init__(self, *_a, **_k):
            pass

        def chat(self, messages):
            raise RuntimeError("worker 崩溃")

        def close(self):
            pass

    orig = _install_fake_llm(BoomClient)
    try:
        out = pool.vote("题目", master_contribution="主方案的唯一候选")
    finally:
        orch.LLMClient = orig

    assert out["ok"] and out["consensus"]
    assert "胜出方案（@master）" in out["report"]
    assert "主方案的唯一候选" in out["report"]
    assert "未给出可用方案" not in out["report"]


def test_bridge_ask_vote_passes_master_contribution(monkeypatch):
    """bridge ask_vote 透传 master_contribution 给 WorkerPool.vote。"""
    from agent import bridge as br

    captured: dict = {}

    class FakeClient:
        def __init__(self, *_a, **_k):
            pass  # 空 agents → 走兜底 solo，但本用例只关心透传，不会真调

        def chat(self, messages):
            return {"role": "assistant", "content": "1"}

        def close(self):
            pass

    monkeypatch.setattr(br, "LLMClient", FakeClient)
    cfg = Config(_interpolate({
        "agents": [{"name": "st", "base_url": "http://x", "api_key": "k", "models": ["m1"]}],
        "llm": {"base_url": "u", "api_key": "k", "model": "m"},
    }))
    b = br.Bridge(cfg)
    orig_vote = b.workers.vote

    def spy_vote(prompt, workers=None, threshold=None, timeout=None, master_contribution=None):
        captured["master_contribution"] = master_contribution
        return {"ok": True, "consensus": True, "report": "ok"}

    b.workers.vote = spy_vote  # 替换实例方法，只验证透传
    try:
        b.handle({"cmd": "ask_vote", "prompt": "题目", "master_contribution": "主方案"})
    finally:
        b.workers.vote = orig_vote
        b.close()

    assert captured.get("master_contribution") == "主方案"
