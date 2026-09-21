"""遗留问题收尾的回归测试：全部离线，无需 API key / 网络。

覆盖（2026-09-21 遗留清单）：
- 默认会话 id 同秒碰撞（旧实现同秒内多个会话共用同一 JSONL 文件）
- 连接池释放：LLMClient.close / WorkerPool.close / Agent.close 必须显式释放，
  不能只靠 GC（Web 频繁建/淘汰会话、bridge 常驻会积压废弃连接）
"""
from __future__ import annotations

from unittest.mock import MagicMock

import agent.core.agent as agent_mod
import agent.core.orchestrator as orch
import agent.core.session as sess_mod
from agent.config import Config, _interpolate
from agent.core.agent import Agent
from agent.core.llm import LLMClient
from agent.core.orchestrator import WorkerPool
from agent.core.session import Session, sanitize_session_id


def _isolate_sessions(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sess_mod, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(sess_mod, "TRASH_DIR", tmp_path / "trash")


# ---------------- 遗留1：默认会话 id 不能同秒碰撞 ----------------

def test_default_session_id_does_not_collide(tmp_path, monkeypatch):
    """同一秒内创建的多个会话必须拿到互不相同的 id。

    实测修正前 `Session()` 连续三次返回完全相同的 id（秒级时间戳），
    三者共用同一个 JSONL：历史互相穿插，Session.cleanup 按文件清理时
    还会一次带走多个会话。
    """
    _isolate_sessions(tmp_path, monkeypatch)
    sessions = [Session() for _ in range(8)]
    ids = [s.session_id for s in sessions]
    assert len(set(ids)) == 8, f"会话 id 发生碰撞：{ids}"
    # 生成的 id 必须仍是安全文件名（可直接落盘）
    for sid in ids:
        assert sanitize_session_id(sid) == sid, sid
    assert len({s._path for s in sessions}) == 8


def test_default_session_id_shape(tmp_path, monkeypatch):
    """默认 id 形如 <时间戳>-<4位随机后缀>，便于按时间肉眼排序。"""
    _isolate_sessions(tmp_path, monkeypatch)
    sid = Session().session_id
    stamp, _, suffix = sid.rpartition("-")
    assert len(stamp) == 15 and stamp[8] == "-", sid          # YYYYmmdd-HHMMSS
    assert len(suffix) == 4 and all(c in "0123456789abcdef" for c in suffix), sid


# ---------------- 遗留2：连接池必须显式释放 ----------------

def test_llm_client_close_is_idempotent():
    """LLMClient.close 释放底层客户端，且重复调用不抛异常。"""
    client = LLMClient("http://x", "k", "m")
    fake = MagicMock()
    client.client = fake
    client.close()
    client.close()
    assert fake.close.call_count == 2


def test_llm_client_close_swallows_errors():
    """释放失败不得向上抛（收尾路径不该被资源释放问题打断）。"""
    client = LLMClient("http://x", "k", "m")
    fake = MagicMock()
    fake.close.side_effect = RuntimeError("already closed")
    client.client = fake
    client.close()  # 不抛即通过


def test_worker_pool_close_releases_all_clients():
    """WorkerPool.close 关闭并清空全部工人客户端，且幂等。"""
    cfg = Config(_interpolate({
        "agents": [{"name": "st", "base_url": "u", "api_key": "k", "models": ["m1", "m2"]}],
    }))
    pool = WorkerPool(cfg, exclude=None)
    c1, c2 = MagicMock(), MagicMock()
    pool._clients = {"st:m1": c1, "st:m2": c2}

    pool.close()
    c1.close.assert_called_once()
    c2.close.assert_called_once()
    assert pool._clients == {}
    pool.close()  # 二次调用不抛


def test_agent_close_releases_main_and_worker_clients(tmp_path, monkeypatch):
    """Agent.close() 要同时释放主模型与自有工人池的连接池。

    实测修正前 Agent.close 只刷会话 + 关 MCP，主模型与工人池的 httpx 连接池
    只能等 GC 回收。
    """
    _isolate_sessions(tmp_path, monkeypatch)
    made: list = []

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            self.closed = 0
            made.append(self)

        def close(self):
            self.closed += 1

    monkeypatch.setattr(agent_mod, "LLMClient", FakeLLM)
    monkeypatch.setattr(orch, "LLMClient", FakeLLM)

    # 一站两模型：主取 st:m1，工人池里就有 st:m2，池才会被创建
    cfg = Config(_interpolate({
        "agents": [{"name": "st", "base_url": "u", "api_key": "k", "models": ["m1", "m2"]}],
        "session": {"flush_batch": 1, "flush_interval": 0.0},
    }))
    bot = Agent(cfg, profile=cfg.profile("st:m1"), enable_workers=True)
    pool = bot.worker_pool
    assert pool is not None, "池里应还有 st:m2 这名工人"

    # 触发工人客户端懒创建
    pool._client_for(cfg.profile("st:m2"))
    assert pool._clients, "应先创建出工人客户端"

    bot.close()

    assert bot.llm.closed == 1, "主模型连接池应被释放"
    assert pool._clients == {}, "工人池客户端应被清空"
    assert all(c.closed == 1 for c in made), [c.closed for c in made]
