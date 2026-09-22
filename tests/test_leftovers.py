"""遗留问题收尾的回归测试：全部离线，无需 API key / 网络。

覆盖：
- 连接池释放：LLMClient.close / WorkerPool.close 必须显式释放，不能只靠 GC
  （bridge 常驻会积压废弃连接）。
- run_shell 参数模式遇 cmd 内置命令（echo/dir/type）失败回退 shell=True；超时不重试。

（原「默认会话 id 防碰撞」「Agent.close 释放」用例已随内部 Agent / 会话持久化模块移除。）
"""
from __future__ import annotations

from unittest.mock import MagicMock

from agent.config import Config, _interpolate
from agent.core.llm import LLMClient
from agent.core.orchestrator import WorkerPool

# ---------------- 连接池必须显式释放 ----------------

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


# ---------------- cmd 内置命令：参数模式失败必须退回 shell ----------------

def test_run_shell_falls_back_to_shell_for_builtins(monkeypatch):
    """参数模式遇 FileNotFoundError（内置命令）时应回退 shell=True 重跑一次。"""
    import agent.tools.builtin as bi

    calls: list[tuple] = []

    class FakeProc:
        stdout = b"ok"
        stderr = b""
        returncode = 0

    def fake_run(cmd, **kw):
        calls.append((cmd, kw.get("shell")))
        if kw.get("shell") is False:
            raise FileNotFoundError("[WinError 2] 系统找不到指定的文件。")
        return FakeProc()

    monkeypatch.setattr(bi.subprocess, "run", fake_run)
    reg = bi.build_builtin_tools(Config(_interpolate({"agents": []})))
    out = reg.execute("run_shell", {"command": "echo hi"})

    assert len(calls) == 2, f"应参数模式失败后回退 shell，实际调用 {calls}"
    assert calls[0][1] is False and calls[1][1] is True, calls
    assert "ok" in out and "[exit code] 0" in out


def test_run_shell_timeout_does_not_retry(monkeypatch):
    """只有"文件找不到"才回退；超时不重试（别把一次挂死变成两次）。"""
    import agent.tools.builtin as bi

    calls: list = []

    def fake_run(cmd, **kw):
        calls.append(kw.get("shell"))
        raise bi.subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(bi.subprocess, "run", fake_run)
    reg = bi.build_builtin_tools(Config(_interpolate({"agents": []})))
    out = reg.execute("run_shell", {"command": "sleep 999"})

    assert "超过" in out and "未完成" in out
    assert len(calls) == 1, f"超时不应重试，实际 {calls}"


def test_run_shell_real_cmd_builtins():
    """真实子进程：cmd 内置命令（echo）必须能跑通并带回退出码。"""
    from agent.tools.builtin import build_builtin_tools

    reg = build_builtin_tools(Config(_interpolate({"agents": []})))
    out = reg.execute("run_shell", {"command": "echo BUILTIN_E2E_OK"})
    assert "BUILTIN_E2E_OK" in out, out[:300]
    assert "[exit code] 0" in out, out[:300]
