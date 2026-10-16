"""第五轮审计（2026-10-15 多模型对撞）修复的回归保护：协议层接线与共享组。

盯住三条"内核有、外壳没接"与一条"并发下拿组即崩"：
- ping / list_agents / ready 必须报**活池**（热同步只改工人池，静态配置不会跟着变）；
- `timeout` / `tags` 是内核一直支持的参数，协议层原先从不透传（文档却写了 tags）；
- `McpGroup` 被上一个使用者释放关闭后，并发取组不得抛 RuntimeError。
全部离线：不碰网络、不需要 key。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.bridge import Bridge, _norm_timeout
from agent.config import Config, _interpolate
from agent.core.orchestrator import WorkerPool
from agent.tools import mcp_client


def _bridge() -> Bridge:
    data = {
        "agents": [{"name": "st", "base_url": "u1", "api_key": "k", "models": ["m1", "m2"]}],
        "llm": {"timeout": 5, "max_retries": 0},
    }
    return Bridge(Config(_interpolate(data), raw=data))


def test_list_agents_and_ping_report_the_live_pool():
    """热同步改了工人池之后，对外清单必须跟着变（旧写法读 cfg，主会点到不存在的名字）。"""
    b = _bridge()
    try:
        assert (
            sorted(b.handle({"cmd": "list_agents"})["agents"][0].keys())
            and len(b.handle({"cmd": "list_agents"})["agents"]) == 2
        )
        del b.workers.profiles["st:m2"]
        names = {a["name"] for a in b.handle({"cmd": "list_agents"})["agents"]}
        assert names == {"st:m1"}, f"list_agents 仍在报静态配置：{names}"
        assert b.handle({"cmd": "ping"})["agents"] == ["st:m1"], "ping 同样必须报活池"
    finally:
        b.close()


def test_ask_many_passes_timeout_and_tags_to_the_kernel():
    """外部主传的 timeout/tags 必须真的到达内核（曾经被静默丢弃）。"""
    b = _bridge()
    seen = {}

    def recorder(workers, prompt, system=None, timeout=None):
        seen["timeout"] = timeout
        seen["workers"] = workers
        return []

    b.workers.ask_many_structured = recorder
    b.workers.pick = lambda limit=None, tags=None: (seen.__setitem__("tags", tags), ["st:m1"])[1]
    try:
        b.handle({"cmd": "ask_many", "prompt": "任务", "timeout": 30, "tags": "code"})
        assert seen["timeout"] == 30
        assert seen["tags"] == "code", "缺省点名时按 tags 选路必须透传"
    finally:
        b.close()


@pytest.mark.parametrize(
    "raw,expected",
    [(None, None), ("", None), (0, None), ("0", None), (-5, None), ("45", 45.0), (20, 20.0)],
)
def test_timeout_is_normalized_before_reaching_the_kernel(raw, expected):
    """`0` 不是"不限"而是"限时 0 秒"（会把全员判未完成），非法类型会让内核抛 TypeError。"""
    assert _norm_timeout(raw) == expected


def test_acquire_group_rebuilds_a_closed_group_instead_of_raising(monkeypatch):
    """并发把共享组用到引用归零后，后来者要能重建，而不是在 ready 行之前就崩。"""
    servers = [{"name": "s1", "transport": "stdio", "command": "noop"}]
    key = mcp_client._group_key(servers)
    closed = mcp_client.McpGroup(servers)
    closed.start = lambda: None
    closed.try_acquire()
    closed._closed = True
    closed._refs = 0
    mcp_client._groups[key] = closed

    built = []

    class Fresh(mcp_client.McpGroup):
        def __init__(self, cfgs):
            super().__init__(cfgs)
            built.append(self)

        def start(self):
            self.conns = []
            self.tools = []

    monkeypatch.setattr(mcp_client, "McpGroup", Fresh)
    group = mcp_client.acquire_group(servers)
    try:
        assert group is not closed, "已关闭的组不能被复用"
        assert group.refs == 1 and built, "应重建一份并登记一个使用者"
        assert mcp_client._groups.get(key) is group, "陈旧条目必须被替换掉"
    finally:
        mcp_client.release_group(group)
        mcp_client._groups.pop(key, None)


@pytest.mark.parametrize(
    "command",
    [
        "echo a>nul\ndel /f /q x.txt",  # cmd 把换行当分隔符，首词只剩 echo
        "de^l /f /q x.txt",  # cmd 执行时 ^ 被剥掉，还原成 del
        "echo hi & powershell -enc AAAA=",  # 首词是 echo，PS 分支根本没进
        'cmd /c "powershell -enc AAAA="',  # 正文藏在引号里，token 循环认不出 -enc
        r"echo x > %USERPROFILE%\start.bat",  # 重定向落点要展开后才知道
        r"echo x > $env:TEMP\y",
    ],
)
def test_shell_safety_blocks_the_six_bypasses(command):
    from agent.tools.shell_safety import _check_command_safety

    assert _check_command_safety(command), f"实测可绕的写法必须被拦：{command!r}"


@pytest.mark.parametrize(
    "command",
    ["dir", "echo hello > out.txt", "ruff format .", "git log --format=%H", 'python -c "print(1)"'],
)
def test_shell_safety_still_allows_normal_commands(command):
    """拦得住，也不能把正常开发命令误杀（本项目曾因"任意位置匹配"大面积误拦）。"""
    from agent.tools.shell_safety import _check_command_safety

    assert _check_command_safety(command) is None


def test_run_script_marks_nonzero_exit_as_failure(monkeypatch, tmp_path):
    """技能脚本非 0 退出必须判失败：旧写法只看 stdout，崩溃也回 ok=True。"""
    import subprocess as sp

    from agent import skills_manager as sm

    mgr = sm.SkillManager(tmp_path)
    skill = tmp_path / "s1"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: s1\ndescription: d\n---\n正文\n", encoding="utf-8")
    (skill / "scripts" / "boom.py").write_text("import sys; sys.exit(2)\n", encoding="utf-8")
    mgr.scan()

    class Proc:
        stdout = b"partial"
        stderr = b"boom"
        returncode = 2

    monkeypatch.setattr(sp, "run", lambda *a, **kw: Proc())
    res = mgr.run_script("s1", "boom.py", {})
    assert getattr(res, "ok", True) is False, "非 0 退出不得被当成成功"
    assert "exit code] 2" in str(res)


def test_vote_names_the_workers_it_had_to_drop(tmp_path):
    """点名的投票人里有不存在的名字时，必须在报告里点名（collect 会静默丢掉它们）。"""
    from agent.core.orchestrator import WorkerPool

    data = {
        "agents": [{"name": "st", "base_url": "u", "api_key": "k", "models": ["m1", "m2"]}],
        "llm": {"timeout": 5, "max_retries": 0},
    }
    pool = WorkerPool(Config(_interpolate(data), raw=data))
    pool.collect = lambda *a, **kw: [("st:m1", "方案A")]
    pool._run_ballot = lambda *a, **kw: (1, {1: 1}, 1, 1, 0)
    try:
        got = pool.vote("任务", workers=["st:m1", "ghost"])
        assert got["ok"] and "ghost" in got["report"], got["report"][:120]
        bad = pool.vote("任务", workers=["ghost"])
        assert not bad["ok"] and "ghost" in bad["report"] and "没有可投票" in bad["report"]
    finally:
        pool.close()


def test_dispatch_thread_count_follows_concurrency_not_fanout():
    """并行派工的线程数必须跟着并发上限走，而不是跟着点名人头数线性增长。

    旧实现给每个工人都起一条 daemon 线程、只靠信号量排队：实测点名 40 个工人时
    peak live threads = 40（每条约 8MB 栈），池子被自动发现撑大之后先撞系统线程上限。
    """
    import threading
    import time as _time

    from agent.core.orchestrator import WorkerPool

    n, cap = 40, 3
    data = {
        "agents": [
            {"name": f"p{i}", "base_url": f"http://y{i}", "api_key": "k", "model": "x"}
            for i in range(n)
        ],
        "collaboration": {"max_workers": cap, "max_per_station": 1, "latency_min_samples": 1},
        "llm": {"timeout": 5, "max_retries": 0},
    }
    cfg = Config(_interpolate(data), raw=data)
    pool = WorkerPool(cfg)
    base = threading.active_count()
    peak = [0]

    def slow_ask(worker, prompt, system=None, memo=None):
        peak[0] = max(peak[0], threading.active_count() - base)
        _time.sleep(0.05)
        return {"worker": worker, "ok": True, "status": "ok", "answer": "x", "error": ""}

    pool.ask_result = slow_ask
    try:
        recs = pool.ask_many_structured(list(pool.profiles), "任务")
        assert len(recs) == n, "点名的工人都必须被派到（语义不变）"
        assert peak[0] <= cap, f"线程数应受并发上限约束：peak={peak[0]} cap={cap}"
    finally:
        pool.close()


def test_worst_case_per_request_counts_retries_and_backoff():
    from agent.core.llm import worst_case_per_request

    assert worst_case_per_request(120.0, 2) == 3 * 120 + 2 * 61.0
    assert worst_case_per_request(120.0, 0) == 120.0, "不重试时就是单次 timeout"


def test_backoff_sleep_happens_outside_the_station_gate(monkeypatch):
    """退避睡眠绝不能占着同站串行门——否则一个坏模型把整站锁住分钟级。

    旧写法把 `LLMClient.chat()`（内部含重试 + `sleep`）整段包在 `_chat_gated` 的门里，
    "一站同时只跑一个模型"于是变成"一站同时只跑一条重试链"：实测同站第二个工人白等
    1.55s，等几乎都是别人的 sleep。现在门只包一次 `chat_once`，重试环在门外跑。
    断言用事件序列（不靠墙钟阈值，Windows 与 ubuntu 同样稳定）。
    """
    import agent.core.orchestrator as orch
    from agent.core.llm import LLMError
    from agent.core.orchestrator import WorkerPool

    class Boom:
        def __init__(self, *a, **kw):
            pass

        def chat_once(self, messages):
            raise LLMError("rate_limit", "429", retryable=True)

        def close(self):
            pass

    data = {
        "agents": [{"name": "st", "base_url": "u", "api_key": "k", "model": "m1"}],
        "llm": {"timeout": 5, "max_retries": 2},
        "collaboration": {"latency_probe": False, "free_node_discovery": False},
    }
    pool = WorkerPool(Config(_interpolate(data), raw=data))
    events: list[str] = []
    real_gate = pool._station_gate

    class Tracked:
        """包一层站门，把取门/放门记进序列（探测用的 acquire(timeout) 也记）。"""

        def __init__(self, sem):
            self._sem = sem

        def __enter__(self):
            events.append("acquire")
            self._sem.acquire()
            return self

        def __exit__(self, *exc):
            events.append("release")
            self._sem.release()
            return False

        def acquire(self, timeout=None):
            events.append("acquire")
            return self._sem.acquire(timeout)

        def release(self):
            events.append("release")
            self._sem.release()

    monkeypatch.setattr(orch, "LLMClient", Boom)
    monkeypatch.setattr(pool, "_station_gate", lambda station: Tracked(real_gate(station)))
    monkeypatch.setattr(orch.time, "sleep", lambda s: events.append("sleep"))
    try:
        r = pool.ask_result("st", "任务")
        assert r["status"] == "error"
        assert events == [
            "acquire",
            "release",
            "sleep",  # 第 1 次尝试后：门已放，才睡
            "acquire",
            "release",
            "sleep",  # 第 2 次尝试后：同上
            "acquire",
            "release",  # 第 3 次（额度用尽）→ 交回上层
        ], f"退避应当发生在门外：{events}"
    finally:
        pool.close()


def test_list_dir_survives_an_entry_vanishing_mid_scan(tmp_path, monkeypatch):
    """条目在遍历中途 stat 失败（被删/被移/无权限）不该把整次列目录打成一个异常。

    旧写法只在外层接 PermissionError，`item.stat()` 抛 FileNotFoundError 会逃到注册表兜底，
    外部主收到的是看不懂的栈而不是目录清单。
    """
    from pathlib import Path

    import agent.tools.builtin as bi

    (tmp_path / "keep.txt").write_text("a", encoding="utf-8")
    (tmp_path / "ghost.bin").write_text("b", encoding="utf-8")
    real_stat = Path.stat

    def flaky_stat(self, *a, **k):
        if self.name == "ghost.bin":
            raise FileNotFoundError(2, "gone")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    cfg = Config(_interpolate({}))
    reg = bi.build_builtin_tools(cfg)
    bi.set_workspace_roots([str(tmp_path)])
    try:
        out = reg.run("list_dir", {"path": str(tmp_path)})
        assert out["ok"] is True, out["result"]
        assert "keep.txt" in out["result"] and "ghost.bin" in out["result"], out["result"]
    finally:
        bi.set_workspace_roots([])


def test_backup_names_do_not_collide_across_shapes(tmp_path):
    """`a/b.md` 与根下真实的 `a__b.md` 不能生成同一个留档名（旧写法把 / 换成 __ 后撞名）。

    只断言"两个名字不同"是假守卫：名字里还有毫秒时间戳与线程号，任何两次写都必然不同名。
    所以这里直接断言**路径哈希段**——把它换成常量，深层与平铺的 `rel` 就重新撞在一起。
    """
    import hashlib
    import re

    import agent.tools.builtin as bi

    root = tmp_path / "proj"
    (root / "a").mkdir(parents=True)
    targets = [root / "a" / "b.md", root / "a__b.md"]
    for t in targets:
        t.write_text("原始内容", encoding="utf-8")
    cfg = Config(_interpolate({}))
    reg = bi.build_builtin_tools(cfg)
    bi.set_workspace_roots([str(root)])
    try:
        names = []
        for t in targets:
            res = reg.run("write_file", {"path": str(t), "content": "改写"})["result"]
            m = re.search(r"留档到 (\S+?)，", res)
            assert m, res
            name = Path(m.group(1)).name
            names.append(name)
            expect = hashlib.sha1(str(t).encode("utf-8")).hexdigest()[:8]
            assert f".{expect}." in name, f"留档名必须带该文件路径的哈希段：{name} 缺 {expect}"
        assert len(set(names)) == 2, names
    finally:
        bi.set_workspace_roots([])


def test_nonpositive_explicit_timeout_falls_back_to_adaptive():
    """显式 `timeout=0` 不该被理解成"限时 0 秒"（那会让全员被判未完成、额度照烧）。"""
    data = {
        "agents": [{"name": "st", "base_url": "u", "api_key": "k", "models": ["m1", "m2"]}],
        "llm": {"timeout": 7, "max_retries": 0},
    }
    pool = WorkerPool(Config(_interpolate(data), raw=data))
    try:
        adaptive = pool._effective_timeout(2, None)
        assert adaptive > 0
        assert pool._effective_timeout(2, 0) == adaptive
        assert pool._effective_timeout(2, -1) == adaptive
        assert pool._effective_timeout(2, 12.5) == 12.5, "正数仍按传入值"
    finally:
        pool.close()
