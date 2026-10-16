"""MAO 冒烟测试：全部离线，无需 API key / 网络。

覆盖：
- 配置解析（load_config）
- 内置工具注册（build_builtin_tools）
- WorkerPool 并发派工不崩（验证数据竞争修复，任务1）
- bridge 协议：真实子进程跑一遍，stdout 必须是纯 JSON 行
- CLI 外壳（python -m agent call）：单发响应的行数/退出码/UTF-8 编码，以及用法错误
  不得污染 stdout
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

from agent.config import load_config
from agent.core.orchestrator import WorkerPool
from agent.tools.builtin import build_builtin_tools


def test_config_loads():
    cfg = load_config()
    assert cfg is not None
    assert isinstance(cfg.agent_profiles, list)
    # collaboration 默认值应可读、合法
    assert cfg.max_workers >= 1


def test_builtin_tools_registered():
    cfg = load_config()
    reg = build_builtin_tools(cfg)
    names = set(reg.names())
    assert len(names) >= 1
    # 至少应注册内置 shell 工具
    assert "run_shell" in names


def test_worker_pool_construction():
    cfg = load_config()
    pool = WorkerPool(cfg, exclude=None)
    # names() 返回当前池内的工人名列表（可能为空的 solo）
    assert isinstance(pool.names(), list)


def test_worker_pool_concurrent_ask_no_crash():
    """并发派工不应因共享状态（LRU 缓存 / 节点健康）竞争而抛异常（任务1）。"""
    cfg = load_config()
    pool = WorkerPool(cfg, exclude=None)
    if not pool.names():
        # 池为空（solo 配置）时仅验证构造与 names() 不崩即可
        assert pool.names() == []
        return

    # 用假 LLMClient 替代真实网络调用
    import agent.core.orchestrator as orch

    fake = MagicMock()
    fake.chat_once.return_value = {"role": "assistant", "content": "ok"}

    def make_fake_client(*_args, **_kwargs):
        return fake

    orig = orch.LLMClient
    orch.LLMClient = make_fake_client
    try:
        worker = pool.names()[0]
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(pool.ask, worker, f"task-{i}") for i in range(20)]
            for f in futs:
                assert f.result(timeout=10) == "ok"
        # 并发写入后，缓存命中仍返回一致结果（验证锁保护了 LRU）
        assert pool.ask(worker, "task-0") == "ok"
    finally:
        orch.LLMClient = orig


def test_bridge_stdout_is_pure_json_lines():
    """真实子进程验证 bridge 协议：stdout 每一行都必须是可解析的 JSON。

    实测修正前，配置缺失变量的 `[配置警告]` 会 print 到 stdout，与 ready/响应行
    混在一起，按行 json.loads 的外部智能体驱动方在启动阶段就会解析失败。
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "-m", "agent", "bridge"],
        input=json.dumps({"cmd": "ping"}, ensure_ascii=False) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(root),
        timeout=120,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines, f"bridge 无输出；stderr={proc.stderr[:400]}"

    parsed = []
    for ln in lines:
        try:
            parsed.append(json.loads(ln))
        except json.JSONDecodeError as e:  # noqa: PERF203
            raise AssertionError(f"stdout 混入非 JSON 行：{ln!r}") from e

    # 首行是 ready 事件，末行是 ping 的响应
    assert parsed[0].get("event") == "ready"
    assert parsed[-1].get("ok") is True


# ---------------- CLI 外壳（python -m agent ...）----------------
# 三条外壳共用同一个 Bridge 内核，但 CLI 这层的参数解析/退出码/编码此前只有 bridge 被测到。
# 子进程跑（不在进程内调 main()）：既验证 `python -m agent` 这条真实入口，也验证 Windows
# 重定向流的 GBK 陷阱——reconfigure_streams 没生效时下面按 UTF-8 的严格解码就会炸。


def _run_cli(*args: str, stdin_text: str = ""):
    import subprocess
    import sys
    from pathlib import Path

    return subprocess.run(
        [sys.executable, "-m", "agent", *args],
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",  # 输出不是合法 UTF-8 就直接抛，别让 GBK 字节混过断言
        cwd=str(Path(__file__).resolve().parent.parent),
        timeout=180,
    )


def test_cli_call_reads_payload_from_stdin_and_prints_one_json_line():
    """`call <cmd>` 省略参数时从 stdin 读一行：输出必须恰好一行 JSON、退出码 0。"""
    import json

    proc = _run_cli("call", "ping", stdin_text=json.dumps({"ok_marker": "ping"}) + "\n")
    assert proc.returncode == 0, f"rc={proc.returncode} stderr={proc.stderr[:400]}"
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"单发调用应只输出一行响应：{lines}"
    resp = json.loads(lines[0])
    assert resp.get("ok") is True, f"ping 应成功：{resp}"


def test_cli_call_reports_bad_payload_as_json_and_exits_2():
    """参数不是合法 JSON：仍然只往 stdout 打一行 JSON 错误、退出码 2（脚本方按 rc 分支）。"""
    import json

    proc = _run_cli("call", "ping", "{not json")
    assert proc.returncode == 2, f"rc={proc.returncode} stderr={proc.stderr[:400]}"
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"错误路径也只能有一行 JSON：{lines}"
    resp = json.loads(lines[0])
    assert resp.get("ok") is False and "解析失败" in resp.get("error", ""), resp


def test_cli_without_subcommand_fails_fast():
    """没给子命令：argparse 直接拒绝（required=True），且不得往 stdout 吐任何非 JSON 内容。"""
    proc = _run_cli()
    assert proc.returncode != 0, "缺子命令应快速失败，而不是静等 stdin"
    assert proc.stdout.strip() == "", f"用法错误不该写 stdout：{proc.stdout[:200]!r}"
    assert "usage" in proc.stderr.lower(), f"stderr 应给出用法：{proc.stderr[:200]}"
