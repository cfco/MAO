"""第八轮审计（2026-10-29）：工具层密钥外流与 shell 包装绕过。

背景：这一轮三条 P0 都落在"安全边界声明与实际控制面不一致"上：
1. `read_file` 早就不外发 `.env`，但 `type .env` / `cat .env` 走 run_shell——首词不在
   任何一张危险词表里，shell 内置命令又必然回落 shell=True，一条命令把全部 key 读进
   stdout 原样外发给外部主（另一个模型）；
2. cmd 包装只认 `cmd /c` 的裸拼写，引号（`"cmd" /c "del x"`）、全路径
   （`C:\\Windows\\System32\\cmd.exe /c del x`）、`%COMSPEC%` 与 POSIX 的 `sh -c "rm -rf x"`
   都把破坏命令留在内层不被扫描——整条黑名单形同虚设；
3. 重定向越界检查要求操作符前有空白，`echo a>..\\..\\evil` 少一个空格即写出项目根。

外加同批修的一条：异常文案不抹 URL 的 userinfo（`https://user:pass@host/v1` 无 `?`
时整段外流）。

全部用例离线、无网络。
"""

from __future__ import annotations

import sys

import pytest

from agent.config import Config, _interpolate
from agent.core.redact import safe_error_text
from agent.tools.builtin import build_builtin_tools
from agent.tools.shell_safety import _check_command_safety


def _cfg(**tools) -> Config:
    data: dict = {"agents": []}
    if tools:
        data["tools"] = tools
    return Config(_interpolate(data))


# ---------------- 1. 命令 Runner 包装：内层必须递归扫描 ----------------


def _blocked(command: str) -> bool:
    return _check_command_safety(command) is not None


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c "del /f /q a.txt"',  # 基线：这条本来就拦得住
        '"cmd" /c "del /f /q a.txt"',  # 外层引号：首词解析成 '"cmd"'，旧写法不再剥包装
        "c:\\windows\\system32\\cmd.exe /c del /f /q a.txt",  # 全路径
        "%COMSPEC% /c del /f /q a.txt",  # 环境变量指向 cmd
        "CMD /K RD /S /Q D:\\x",  # 大小写 + /k
        'cmd /c "de^l /f /q a.txt"',  # 正文里的 ^ 转义（cmd 执行时还原成 del）
        'sh -c "rm -rf /tmp/x"',  # POSIX 同款包装：旧写法把 sh 当普通首词放过
        'bash -lc "del /f /q x"',  # 短选项簇 -lc
        'cmd /c "echo hi & del /f /q a.txt"',  # 包装内再串联
        '& %comspec% /c "del /f /q x.txt"',  # PowerShell 调用运算符 + 环境变量
        'cmd /c "rm -rf /var/tmp/x"',  # Windows 包装里套 POSIX 破坏词
        "sh -c 'del /f /q x'",  # POSIX 包装里套 Windows 破坏词
        "cmd /c \"python -c 'import ctypes'\"",  # 包装里的内联代码：高危 API 要扫到
        "sh -c \"node -e 'child_process.exec(1)'\"",  # 同款：POSIX 包装 + node 高危 API
    ],
)
def test_wrapped_dangerous_command_blocked(command):
    """包装/变体写法的内层破坏命令必须被扫到（修复前全部放行）。"""
    assert _blocked(command), f"应拦截却放行：{command!r}"


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c "dir"',
        'cmd /c "echo hello"',
        "cmd /c copy a.txt b.txt",  # copy 不在危险词表
        'sh -c "ls -la"',
        "python -m pytest -q",
        "git log --oneline",
        "ruff format .",
        "for %f in (a.txt) do @echo %f",  # for 的正常用法
    ],
)
def test_wrapped_normal_commands_still_allowed(command):
    """反向：递归扫描内层不能把正常包装命令一并误杀。"""
    assert not _blocked(command), f"应放行却被拦：{command!r}"


# ---------------- 2. 重定向越界：不留"少一个空格"的口子 ----------------


@pytest.mark.parametrize(
    "command",
    [
        "echo a>..\\..\\evil.txt",  # 修复前：操作符前无空白 → 整条不匹配
        "echo a >../../evil.txt",
        "echo x>../../Windows/System32/drivers/etc/hosts",
        "echo x>>..\\..\\evil.txt",
        "echo x 2>..\\..\\evil.txt",
    ],
)
def test_redirect_without_space_still_checked(command):
    """重定向目标越界与"操作符前有没有空格"无关。"""
    assert _blocked(command), f"应拦截却放行：{command!r}"


@pytest.mark.parametrize(
    "command",
    [
        "echo hello > out.txt",
        "echo hello >> out.txt",
        "python x.py 2>&1",  # fd 引用不是路径
        "echo x > nul",  # 设备名
        "sort -t'>' -k2 data.txt",  # 引号内的 > 是字面量
        "git show HEAD^ --format=%H",  # 不含重定向操作符的正常命令
    ],
)
def test_harmless_redirect_and_operators_allowed(command):
    """反向：收紧重定向正则不得误拦正常写法与比较/箭头符号。"""
    assert not _blocked(command), f"应放行却被拦：{command!r}"


# ---------------- 3. 密钥文件：shell 通道与文件通道同一口径 ----------------


@pytest.mark.parametrize(
    "command",
    [
        "type .env",  # Windows shell 内置：read_file 拦了，shell 侧原先整份外发
        "cat .env",
        "more < .env",
        "head -5 .env",
        "cp .env /tmp/x",
        "echo sk-abcdef123456 >> .env",
        "echo k>..\\..\\.env",
        "python -c \"print(open('.env').read())\"",
        "cat config/secrets.json",
    ],
)
def test_secret_file_via_shell_blocked(command):
    """命令正文出现密钥文件名一律拒，读写两个方向都拦。"""
    assert _blocked(command), f"应拦截却放行：{command!r}"


@pytest.mark.parametrize(
    "command",
    [
        "cp .env.example .env.example.bak",  # 模板可读写，且不能被 .env 前缀误伤
        "cat .env.example",
        "python -c \"print(open('environment.txt').read())\"",  # myenv 类同形词
        "cat README.md",
    ],
)
def test_non_secret_lookalikes_allowed(command):
    """反向：`.env.example` 与同形词不得被 dotenv 判定误拦。"""
    assert not _blocked(command), f"应放行却被拦：{command!r}"


def test_write_and_append_tools_refuse_secret_files(tmp_path, monkeypatch):
    """文件通道的写入侧与读取侧同一口径（覆写 .env = 把 key 发往外部端点）。

    密钥文件一律用 **tmp_path 里的假 .env**（经 tools.workspace 授权后才够得着守卫）：
    这条用例同时是变异锚点 M81 的靶子，变异会把守卫关掉——写在项目根的真 `.env` 上
    就会在那一刻被 `content=` 覆盖掉（2026-10-29 真发生过一次，本机 .env 被抹成一行）。
    所有可能被单点变异"放开"的写路径，用例都必须只写到自己名下的临时目录里。
    """
    target = str(tmp_path / ".env")
    # 经 tools.workspace 把临时目录授权进来（走真实的 set_workspace_roots 路径），
    # 这样 _resolve 能放行、守卫才有机会被触发
    reg = build_builtin_tools(_cfg(workspace=[str(tmp_path)]))
    for tool, args in (
        ("write_file", {"path": target, "content": "NODE_A_KEY=sk-x"}),
        ("append_file", {"path": target, "content": "NODE_A_KEY=sk-x"}),
        ("read_file", {"path": target}),
    ):
        out = str(reg.execute(tool, args))
        assert "密钥文件" in out, f"{tool} 未拦：{out}"
    assert not (tmp_path / ".env").exists(), "守卫拦下后仍不该落盘"
    # 模板仍可读写：口径收紧不能把 .env.example 一起禁掉
    (tmp_path / ".env.example").write_text("NODE_A_ENDPOINT=\n", encoding="utf-8")
    assert "密钥文件" not in str(reg.execute("read_file", {"path": str(tmp_path / ".env.example")}))


# ---------------- 4. run_shell 输出过凭证掩码 ----------------


def test_run_shell_output_masks_secret_shaped_lines():
    """stdout 不再原样外发：形如 `*_KEY=<长值>` 的行必须被掩码。

    绕过 `_child_env` 的通道很多（子进程自己读文件、脚本回显配置），这一层不依赖
    文件名判定，只看输出内容里的凭证形态。
    """
    probe = (
        "print('fail NODE_A_KEY=supersecretvalue CF_API_TOKEN=anothertoken monkey=3 status=404')"
    )
    reg = build_builtin_tools(_cfg())
    out = str(reg.execute("run_shell", {"command": f'"{sys.executable}" -c "{probe}"'}))
    assert "NODE_A_KEY=****" in out, out[:300]
    assert "CF_API_TOKEN=****" in out, out[:300]
    assert "supersecretvalue" not in out and "anothertoken" not in out, out[:300]
    assert "monkey=3" in out, "同尾词不该被误抹（与 safe_error_text 同一口径）"


# ---------------- 5. 异常文案抹 URL userinfo ----------------


def test_safe_error_text_masks_url_userinfo():
    """无 query 段的端点串以前整段外流：`scheme://` 到第一个 `@` 之间必须抹掉。"""
    leaked = safe_error_text("Connection error for https://user:pwd@router.example.com/v1")
    assert "pwd@" not in leaked and "user:" not in leaked, leaked
    assert "router.example.com/v1" in leaked, f"host 要保留供排障：{leaked}"
    # 正常 URL（无 @）与带 query 的 URL 行为不变
    assert safe_error_text("https://api.example.com/v1") == "https://api.example.com/v1"
    assert "?" in safe_error_text("https://api.example.com/v1?key=sk-abcdef123")
