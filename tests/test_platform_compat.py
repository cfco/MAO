"""第七轮：Windows / Ubuntu 双平台兼容回归（2026-10-22）。

背景：CI 跑 ubuntu-latest，但工具层历史上按 Windows cmd 语义设计，双平台各有缺口，
本轮审计（2026-10-22）实测发现三处：
1. shell_safety 的 POSIX 破坏词表只在内联代码正文里生效，顶层 `rm -rf /` 直过——
   与 run_shell 描述声明的"rm/del/format 都拦"契约相悖；
2. run_shell 超时只杀直接子进程，`sh -c 'x &'` / nohup / setsid 的孙进程残留成孤儿；
3. run_shell 工具描述写死"Windows cmd 语法"，Linux 上外部主拿到错误命令示范。

全部用例离线、无网络。跨平台断言一律按 sys.platform 分侧或 monkeypatch 平台，
保证 Windows 本机与 ubuntu CI 跑的是同一份断言、都真正被执行。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from agent.config import Config, _interpolate
from agent.tools.builtin import _shell_flavor, build_builtin_tools
from agent.tools.shell_safety import _check_command_safety


def _cfg(**tools) -> Config:
    data: dict = {"agents": []}
    if tools:
        data["tools"] = tools
    return Config(_interpolate(data))


# ---------------- 1. run_shell 描述按平台给正确“方言” ----------------


def test_shell_flavor_switches_with_platform(monkeypatch):
    """风味函数必须按平台分流：monkeypatch 后两个分支在同一进程里都能验。

    不按平台分侧就会有一半分支永远不被执行（常量写法在 Linux 上永远测不到
    Windows 分支，反之亦然）——那种"看起来测了"的用例正是恒真用例。
    """
    import agent.tools.builtin as bi

    monkeypatch.setattr(bi.sys, "platform", "win32")
    win = bi._shell_flavor()
    assert "cmd" in win, win

    monkeypatch.setattr(bi.sys, "platform", "linux")
    posix = bi._shell_flavor()
    assert "POSIX" in posix and "sh" in posix, posix
    assert "cmd 语法" not in posix, "Linux 侧不得再示范 cmd 方言"


def test_run_shell_description_carries_host_flavor():
    """注册给外部主看的描述必须接上本平台风味（build 时真的调了 _shell_flavor）。"""
    reg = build_builtin_tools(_cfg())
    desc = reg.get("run_shell").description
    assert _shell_flavor() in desc, desc[:200]


# ---------------- 2. POSIX 破坏命令顶层必须拦（历史缺口） ----------------


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -rf ~",
        "rm -f important.txt",
        "rmdir /s /q x",  # 两侧词表都收
        "mkfs.ext4 /dev/sda",  # 变体后缀：mkfs 前缀兜底（精确匹配拦不住）
        "mkfs /dev/sdb",
        "mkswap /dev/sda1",
        "dd if=/dev/zero of=/dev/sda",
        "shred -u secret.txt",
        "truncate -s 0 x.txt",
        "poweroff",
        "halt",
        "sfdisk /dev/sda",
        "wipefs -a /dev/sda",
        "find . -name x -delete",  # find 的 -delete 等价递归删，按 rm 拦
        "find . -exec rm {} +",  # find -exec 的目标词不在首词位置
        "ls | xargs rm -f",  # 单管道串 xargs：首词是 xargs，破坏词在参数位
        'bash -c "find . -delete"',  # 内联正文里的 Runner 同样要拦
        'sh -c "ls | xargs rm -rf ~"',
    ],
)
def test_posix_dangerous_commands_blocked_at_top_level(command):
    """Linux 原生破坏命令落在顶层（非内联正文）也必须拦。

    修复前这些全部放行：顶层循环只查 _CMD_DANGEROUS_SET（Windows 词表），
    POSIX 词表仅在内联代码正文里生效。CI 跑 ubuntu 却没踩到，因为旧测试
    只覆盖了 `bash -c "rm -rf ~"` 这类内联场景。
    """
    assert _check_command_safety(command), f"应拦截却放行：{command!r}"


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "cat file.txt",
        "grep -rn pattern .",
        "python -m pytest -q",
        "make clean",
        "ruff format .",
        "git status",
        "echo hello",
        "find . -name '*.py' | wc -l",  # find + 单管道的正常用法
        "find . -name '*.pyc' -exec grep -l foo {} +",  # -exec 目标非破坏词
        "xargs echo hi",  # xargs 目标非破坏词
        "pip install requests",
        "df -h",
        "chmod +x script.sh",  # chmod 是正常开发命令，不能按"破坏性变体"误拦
        "mkdir -p a/b",
        "tar -czf out.tgz dir",
        'bash -c "make clean && echo ok"',  # 内联正文里的正常命令不受影响
        'sh -c "echo a | wc -l"',
    ],
)
def test_posix_normal_commands_still_allowed(command):
    """反向：补顶层词表不能误杀 Linux 正常开发命令（本项目曾被误拦坑过）。"""
    assert _check_command_safety(command) is None, f"应放行却被拦：{command!r}"


# ---------------- 3. 超时杀进程树（孙进程不留孤儿） ----------------


def _pid_alive(pid: int) -> bool:
    """pid 是否仍活着（跨平台）。

    Windows：OpenProcess(SYNCHRONIZE) 打得开即活着；
    POSIX：/proc/<pid>/stat 的状态字为 Z（被杀待回收的僵尸）视同消亡——
      僵尸的 pid 还占着进程表，直接探活会误判成"没杀掉"；
    无 /proc 的平台（macOS）：退 os.kill(pid, 0) 探活。
    """
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    proc_dir = Path("/proc")
    if not proc_dir.exists():
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    try:
        stat = (proc_dir / str(pid) / "stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    # stat 格式：`pid (comm) state …`，comm 可能含空格/括号，从最后一个 ) 后取状态字
    return stat.rsplit(")", 1)[-1].split()[0] != "Z"


def _pid_gone(pid: int, wait_s: float = 5.0) -> bool:
    """轮询等待 pid 消亡：进程回收有短暂窗口，不能一枪定输赢（两侧都不 flaky）。"""
    deadline = time.monotonic() + wait_s
    while True:
        if not _pid_alive(pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def test_run_shell_timeout_kills_process_tree(tmp_path):
    """真实子进程：超时后直接子进程与孙进程都必须死（跨平台验尸）。

    构造：父母 python spawn 一个 sleep 60 的孩子、写下孩子 pid，然后自己 sleep 60。
    shell_timeout=3s 跑之 → 超时 → _kill_process_tree：POSIX killpg 收整组，
    Windows taskkill /T 收整树。任一条路失效，本用例都能抓到（孩子还活着）。
    超时给到 3s 而非 1s：父母解释器要先把 pid 写盘再进入 sleep，Windows 冷启动
    慢时 1s 可能还没写到就超时，会把"没拿到 pid"误报成用例错误。
    """
    child_pid_file = tmp_path / "child.pid"
    parent_py = tmp_path / "parent.py"
    parent_py.write_text(
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"with open({str(child_pid_file)!r}, 'w') as f:\n"
        "    f.write(str(p.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    reg = build_builtin_tools(_cfg(shell_timeout=3))
    out = reg.execute("run_shell", {"command": f'"{sys.executable}" "{parent_py}"'})

    assert "超过" in out and "未完成" in out, out[:300]
    child_pid = int(child_pid_file.read_text(encoding="utf-8").strip())
    assert _pid_gone(child_pid), f"孙进程 {child_pid} 在超时后仍存活：杀进程树未生效"


def test_run_shell_spawns_children_in_new_session(monkeypatch):
    """杀进程树的前提：spawn 必须带 start_new_session=True（POSIX 的进程组）。

    不验证这一条，上面的验尸用例在 POSIX 上也可能因"组没建起来"而假绿/假红；
    两个平台都查（Windows 上该参数被忽略，但只要传了就不算回归）。
    """
    import agent.tools.builtin as bi

    seen: list[dict] = []

    class FakeProc:
        returncode = 0

        def communicate(self, timeout=None):
            return (b"ok", b"")

    def fake_popen(cmd, **kw):
        seen.append(kw)
        return FakeProc()

    monkeypatch.setattr(bi.subprocess, "Popen", fake_popen)
    reg = build_builtin_tools(_cfg())
    reg.execute("run_shell", {"command": "echo hi"})

    assert seen, "应至少 spawn 一次"
    assert all(kw.get("start_new_session") is True for kw in seen), seen
