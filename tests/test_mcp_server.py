"""MCP server 入口（agent/mcp_server.py）与 one-shot（mao call）的回归测试。

盯两件事：
1) MCP 外壳与 bridge 内核同源——工具表必须与 Bridge.handle 的指令集一一对应，
   调用结果就是 bridge 响应的原样 JSON（本测试用进程内 memory streams，离线可跑）；
2) `mao call` 子进程契约：stdout 恰好一行 JSON，成败用退出码表达（0=ok，1=指令失败，2=参数坏）。

不断言失败响应的具体字段形状（status/error 的契约正由另一批改动收敛），只断言
"透传一致 + 退出码正确"——外壳测试不该把内核的演进细节焊死。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import anyio
import pytest

from agent.config import Config
from agent.mcp_server import build_server

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_TOOLS = {
    "ping", "list_agents", "health", "list_tools", "call_tool", "load_skill",
    "run_skill_script", "ask", "ask_many", "ask_vote", "run_review",
}


def _with_session(go):
    """起一个进程内 MCP server（空配置：无池、无 MCP server、真实 skills 目录），
    用 memory streams 连一个 ClientSession 跑 go(session)。"""
    from mcp import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    app, bridge = build_server(Config({}))

    async def _main():
        async with create_client_server_memory_streams() as (client, server):
            async with anyio.create_task_group() as tg:
                async def _serve():
                    low = app._lowlevel_server  # noqa: SLF001 - SDK 测试惯用入口
                    await low.run(server[0], server[1], low.create_initialization_options())
                tg.start_soon(_serve)
                async with ClientSession(client[0], client[1]) as session:
                    await session.initialize()
                    await go(session)
                tg.cancel_scope.cancel()

    try:
        anyio.run(_main)
    finally:
        bridge.close()


def _payload(result) -> dict:
    """CallToolResult 的首个 text 内容应为 bridge 响应的 JSON 原文。"""
    assert not result.is_error, f"协议层报错: {result.content}"
    return json.loads(result.content[0].text)


def test_tool_table_matches_bridge_commands():
    seen: list[str] = []

    async def go(s):
        tools = await s.list_tools()
        seen.extend(t.name for t in tools.tools)

    _with_session(go)
    assert set(seen) == EXPECTED_TOOLS, "MCP 工具表必须与 bridge 指令集一一对应"


def test_ping_roundtrip_through_mcp():
    got: dict = {}

    async def go(s):
        got.update(_payload(await s.call_tool("ping", {})))

    _with_session(go)
    assert got["ok"] is True and "version" in got


def test_call_tool_reads_project_file():
    """call_tool 透传 + 绝对/相对路径参数不被 MCP 层的安全校验误拦。"""
    got: dict = {}

    async def go(s):
        got.update(_payload(await s.call_tool(
            "call_tool", {"name": "read_file", "args": {"path": "config.yaml"}})))

    _with_session(go)
    assert got["ok"] is True and "agents" in got["result"]


def test_health_on_empty_pool():
    got: dict = {}

    async def go(s):
        got.update(_payload(await s.call_tool("health", {})))

    _with_session(go)
    assert got["ok"] is True and got["workers"] == []


def _run_call(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "agent", "call", *argv],
        capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT), timeout=120,
    )


def test_one_shot_ping_prints_one_json_line():
    proc = _run_call("ping")
    assert proc.returncode == 0, proc.stderr[:400]
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"stdout 必须恰好一行 JSON，实得 {lines}"
    assert json.loads(lines[0])["ok"] is True


def test_one_shot_reads_payload_from_stdin():
    proc = subprocess.run(
        [sys.executable, "-m", "agent", "call", "call_tool"],
        input=json.dumps({"name": "list_dir", "args": {"path": "."}}, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT), timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[:400]
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    resp = json.loads(lines[-1])
    assert resp["ok"] is True and "config.yaml" in resp["result"]


def test_one_shot_unknown_cmd_exits_failed():
    proc = _run_call("definitely_not_a_cmd")
    assert proc.returncode == 1
    assert "未知指令" in proc.stdout


def test_one_shot_bad_json_exits_usage_error():
    proc = _run_call("ask", "{不是json")
    assert proc.returncode == 2
    assert "JSON" in json.loads(proc.stdout)["error"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
