"""Shell 命令安全扫描：黑名单 + 注入模式 + 内联代码防护 + 重定向越界检测。

本模块是 builtin.run_shell 的前置安全检查，独立成文件的原因：
- 原 builtin.py 658 行，单文件超 500 行约定；安全扫描约占 250 行，单独抽出后 builtin.py
  专注工具定义与文件安全，shell_safety.py 专注 shell 命令安全纵深防御，各司其职。
- 安全逻辑将来可能复用到别的执行入口（如 MCP server 里接 shell 命令透传），独立模块
  可以直接 import，不用再从 builtin.py 挖内部函数。

防护三层（见 _check_command_safety 的 docstring）：
  ① 危险命令黑名单（锚定命令首词）+ shell 注入模式拦截；
  ② python -c / node -e 等「解释器内联代码」正文再扫一遍高危 API/内嵌命令/越界写；
  ③ shell=False 优先参数模式，重定向目标也受项目根约束。
启发式不可能穷尽：真要硬隔离得靠 OS/容器层。
"""
from __future__ import annotations

import re
import shlex

# ==============================================================
# run_shell 安全加固：命令黑名单 + 注入模式拦截
# ==============================================================

# Windows cmd 危险命令（不可逆/破坏性）。
# 注意：这里只保留**命令词本身**，匹配时锚定"命令首词"位置（见 _check_command_safety）。
# 早期实现按"任意位置出现该词"匹配，会把参数位的同名词一并拦下：
#   ruff format .（format 作参数）、make clean（clean 作参数）、git log --format=%H
# 这些正常开发命令会被大面积误拦。危险命令只有被当作命令执行（首词）时才有破坏力，
# 因此改为首词匹配：`format D:`、`del /f /q /s x` 照拦，`ruff format .` 放行。
_CMD_DANGEROUS = [
    "format",        # 格式化磁盘
    "fdisk",         # 分区（旧版 Windows）
    "diskpart",      # 磁盘分区工具（可删除卷）
    "clean",         # diskpart clean 清磁盘
    "chkdsk",        # 修复磁盘（可能强制卸载卷）
    "rd",            # rd /s /q 递归静默删除目录（比 rm -rf 还狠）
    "rmdir",         # 同上
    "del",           # del /f /q /s 强制静默递归删除文件
    "erase",         # 同上
    "shutdown",      # 关机/重启
    "reboot",        # 重启（旧命令）
    "reg",           # reg delete / reg import 删改注册表
    "schtasks",      # schtasks /delete 删除计划任务
    "taskkill",      # taskkill /f /im 强制杀进程
    "wmic",          # wmic process delete 删除进程
    "bootcfg",       # 修改引导配置
    "bcdedit",       # 修改引导配置（Win Vista+）
    "syskey",        # 系统数据库加密（不可逆）
]
_CMD_DANGEROUS_SET = frozenset(_CMD_DANGEROUS)

# cmd /c、cmd /k 包装前缀：剥掉后，被包装的真实命令词才落到首词位置
_CMD_WRAPPER = re.compile(r"^\s*cmd(?:\.exe)?\s*/[ck]\s+", re.IGNORECASE)
# Windows 可执行后缀：format.com / del.exe 等真实二进制要归一到命令词再比对
_EXE_SUFFIXES = (".exe", ".com")
# 命令分隔符：一次请求里可能串多条命令（echo x & del /s y），每段的首词都要查
_CMD_SEPARATOR = re.compile(r"&&|\|\||[&|;]")

# PowerShell 危险命令（带破坏性参数才拦）
_PS_DANGEROUS = [
    (r"Remove-Item\s+.*-Recurse\s+-Force", "PowerShell: Remove-Item -Recurse -Force（递归强删）"),
    (r"Remove-Item\s+.*-Force", "PowerShell: Remove-Item -Force（强删）"),
    (r"Stop-Process\s+.*-Force", "PowerShell: Stop-Process -Force（强杀进程）"),
    (r"Set-ExecutionPolicy\s+.*Bypass", "PowerShell: Set-ExecutionPolicy Bypass（绕过执行策略）"),
    (r"Invoke-WebRequest\s+.*-OutFile\s+.*\.ps1", "PowerShell: Invoke-WebRequest 下载并保存脚本"),
    (r"DownloadString\s+.*\.ps1", "PowerShell: DownloadString 下载脚本（可能执行）"),
    (r"iex\s*\(", "PowerShell: iex() 执行动态代码（Invoke-Expression 别名）"),
    (r"Invoke-Expression", "PowerShell: Invoke-Expression 执行动态代码"),
    (r"Start-Process\s+.*-Verb\s+RunAs", "PowerShell: 以管理员身份启动"),
]

# 全局注入模式：检测 shell 命令拼接（在参数模式下这些拼接不会生效，但用户写了就是意图）
_INJECTION_PATTERNS = [
    re.compile(r"(?:&&|\|\|)\s*\S+"),                      # cmd1 && cmd2 / cmd1 || cmd2
    re.compile(r"[|&]\s*\S+\s*[|&]"),                      # 中间嵌套管道
    re.compile(r";\s*\S+\s*;\s*\S+"),                       # 分号多段（Unix shell 风格，Windows cmd 不识别但 PowerShell 可）
    re.compile(r"\$\([^)]+\)"),                             # 命令替换 $(...)
    re.compile(r"`[^nrt]", re.DOTALL),                      # PowerShell 反引号转义接非控制字符
]

# shell=False 无法支持的 shell 专有语法特征（检测到则退回 shell=True）
_SHELL_ONLY_SIGNS = re.compile(r"[|><&]|\|\||&&")

# 成对引号包裹的内容（单/双引号）
_QUOTED_RE = re.compile(r'"[^"]*"|\'[^\']*\'')


def _strip_wrapping_quotes(token: str) -> str:
    """剥掉 token 首尾成对的包裹引号（"..." 或 '...'）。

    shlex posix=False 分词会保留引号字符：`python -c "print(1)"` 会把
    带引号的表达式当参数传给 Python（静默无输出），带空格的路径同理损坏。
    引号语义已在分词时生效，这里只去掉包裹字符，恢复真实参数值。
    """
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ('"', "'"):
        return token[1:-1]
    return token


def _mask_quoted(command: str) -> str:
    r"""把引号包裹的内容替换为等长占位符，专供注入模式匹配使用。

    为什么：`echo "a|b|c"` 里的管道符是字面量、不是命令拼接，直接对整串匹配会误拦；
    而 `python -c "print(1|2)"` 又因规则要求两侧都得分隔符而放行 —— 同一类写法两种
    结果，行为不一致。先屏蔽引号内容再匹配，`cmd1 && cmd2`、`a | b` 这类真实拼接照拦。

    占位符用 \x00：它不属于 \s，不会与分隔符规则混淆；等长替换（按原串长度补位）
    保证命中位置在原文里能直接取到片段用于提示文案。
    """
    return _QUOTED_RE.sub(lambda m: "\x00" * len(m.group()), command)


# shell 重定向操作符：> >> < 2> 2>> 1> &> 等，后跟目标文件名。
# 用于检测「echo x > ..\..\etc\passwd」这类通过重定向绕过 cwd 限制的写越界。
# 目标捕获：引号包裹的串 或 不含空白/特殊符号的 token。
_REDIRECT_RE = re.compile(
    r"(?:^|\s)"                          # 开头或空白分隔
    r"(?:\d*&?)?"                        # 可选 fd 前缀：1/2/&（1>, 2>>, &>）
    r"(?:>>|>|<)"                        # 操作符：>> 优先于 > 匹配
    r"\s*"
    r"(\"[^\"]*\"|'[^']*'|[^\s&|<>]+)"   # 目标：引号串 或 非特殊 token
)


def _check_redirect_targets(command: str) -> str | None:
    r"""检查 shell 重定向目标是否逃出项目根目录。

    cwd 被限制在项目根内，但重定向符的目标路径不受 cwd 约束：
      echo x > ..\..\Windows\System32\drivers\etc\hosts
    会把内容写到项目根之外。这里对每个重定向目标做路径穿越检测，
    命中即返回拦截说明，安全返回 None。

    注意：这是启发式检查，不解析完整 shell 语法；目标是挡住最常见的越界写法，
    而非实现一个完整的 shell 解析器。设备名（NUL/CON 等）与 fd 引用（&1）放行。
    """
    for m in _REDIRECT_RE.finditer(command):
        target = m.group(1)
        # 剥引号："> \"my file.txt\"" 的目标是带引号的
        target = _strip_wrapping_quotes(target)
        if not target:
            continue
        # fd 引用（>&1, 1>&2）不是文件路径，放行
        if target.startswith("&"):
            continue
        low = target.lower()
        # 设备名放行（NUL/CON/PRN/AUX/COMx/LPTx 等）
        if re.match(r"^(nul|con|prn|aux|com[1-9]|lpt[1-9])(\.|$)", low):
            continue
        # 绝对路径（盘符 C: 或根斜杠）直接越界
        if re.match(r"^[a-zA-Z]:", target) or target.startswith(("/", "\\")):
            return (f"命令拦截：重定向目标 `{target}` 是绝对路径，"
                    f"可能写出项目根目录，已拒绝")
        # 相对路径穿越：含 .. 的目标可能逃出项目根
        if ".." in target.replace("\\", "/").split("/"):
            return (f"命令拦截：重定向目标 `{target}` 含路径穿越(..)，"
                    f"可能写出项目根目录，已拒绝")
    return None


def _head_word(segment: str) -> str:
    """取一段命令中"实际被执行"的命令首词（小写）。

    归一化四件事，避免用拼写差异绕过首词匹配：
    - 剥掉 `cmd /c`、`cmd /k` 包装前缀；
    - 剥掉包裹引号（`"del" /s x` 的引号不改变它是 del 命令）；
    - 只取首词（`"format D:"` 剥引号后含空格，要再切一次才拿到 format）；
    - 去掉路径前缀与 .exe/.com 后缀（`C:\\...\\format.com` → `format`）。
    """
    s = _CMD_WRAPPER.sub("", segment.strip())
    if not s:
        return ""
    try:
        tokens = shlex.split(s, posix=False)
    except ValueError:
        tokens = [s]
    if not tokens:
        return ""
    # 剥引号后可能仍带空格（整个命令被一对引号包住），取第一个词才是命令词
    parts = _strip_wrapping_quotes(tokens[0]).strip().split()
    if not parts:
        return ""
    low = parts[0].replace("\\", "/").rsplit("/", 1)[-1].lower()  # 去路径前缀
    for suf in _EXE_SUFFIXES:
        if low.endswith(suf):
            return low[: -len(suf)]
    return low


def _command_heads(command: str) -> list[str]:
    """把命令按 shell 分隔符切段，返回每段的命令首词（全部要查）。

    只查整条命令的首词会漏掉串联在后面的危险命令：`echo x & del /s y` 的
    首词是 echo，但真正执行删除的是 del。`&&`/`||` 已被注入模式拦下，
    这里的切段是单 `&`、`|`、`;` 串联场景的兜底。
    """
    return [h for h in (_head_word(p) for p in _CMD_SEPARATOR.split(command)) if h]


_PS_PATTERNS = [(re.compile(p, re.IGNORECASE), desc) for p, desc in _PS_DANGEROUS]


# ==============================================================
# 解释器内联代码防护（#5）：python -c / node -e / perl -e … 首词是解释器、
# 破坏逻辑藏在代码正文里，只扫"命令首词"会被整体绕过。这里对"内联代码正文"
# 再扫一遍危险 API / 内嵌 shell 命令 / 重定向越界。纵深防御、非沙箱。
# ==============================================================
_INTERP_INLINE = {
    "python", "python3", "python2", "py", "pypy", "pypy3",
    "node", "nodejs", "deno", "bun",
    "ruby", "perl", "php", "osascript", "tclsh", "wish", "expect",
}
# 携带"内联代码字符串"的开关（powershell/pwsh 的 -Command 走它自己的分支）
_INLINE_CODE_FLAGS = {"-c", "-e", "--eval", "-r", "--rc", "-import", "--import"}
# POSIX 侧破坏性命令（内联代码常调 sh/os.system，补齐 Windows 集合看不到的词）
_POSIX_DANGEROUS_SET = {"rm", "rmdir", "mkfs", "dd", "shred", "mkfstools", "truncate"}
_ALL_DANGEROUS_SET = _CMD_DANGEROUS_SET | _POSIX_DANGEROUS_SET
# 内联代码里的"高危 API"：无 shell 对应词、破坏性/绕过黑名单意图明显
_CODE_HIGH_RISK = [
    (r"\bshutil\s*\.\s*rmtree\b", "shutil.rmtree（递归删除）"),
    (r"\bos\s*\.\s*(remove|unlink|removedirs|kill|killpg|startfile|ftruncate)\b", "os 文件/进程删除"),
    (r"\bos\s*\.\s*execv?[e]?\b", "os.exec*（替换进程执行）"),
    (r"\bpty\s*\.\s*spawn\b", "pty.spawn"),
    (r"\bctypes\b", "ctypes 直连系统 API"),
    (r"__import__\s*\(", "__import__ 动态导入"),
    (r"\beval\s*\(", "eval() 动态求值"),
    (r"\bexec\s*\(", "exec() 动态执行"),
    (r"\bchild_process\b", "Node child_process"),
    (r"\bfs\s*\.\s*(rm|rmdir|unlink)\w*Sync\s*\(", "Node fs 删除"),
    (r"\.Delete\s*\(", ".Delete() 删除"),
]
# 代码里以引号包裹的"内嵌 shell 命令串"（os.system('del ...') 等）
_EMBEDDED_Q = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def _split_tokens(command: str) -> list[str]:
    """尝试把命令拆分成参数列表；失败返回空列表。"""
    try:
        return shlex.split(command, posix=False)
    except ValueError:
        return []


def _extract_inline_code(tokens: list[str]) -> tuple[str | None, str]:
    """若首 token 是内联解释器且带内联代码开关，返回 (解释器名, 代码正文)；否则 (None, "")。"""
    if not tokens:
        return None, ""
    interp = tokens[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    if interp.endswith(".exe"):
        interp = interp[:-4]
    if interp not in _INTERP_INLINE:
        return None, ""
    for i, t in enumerate(tokens[1:], start=1):
        flag = _strip_wrapping_quotes(t).strip().lower()
        if flag in _INLINE_CODE_FLAGS and i + 1 < len(tokens):
            return interp, _strip_wrapping_quotes(tokens[i + 1])
    return None, ""  # 是解释器但无内联代码（如 python script.py：正文不可静态扫描，放行）


def _scan_code_body(body: str) -> str | None:
    """扫描"代码正文 / 内联脚本"里的破坏性内容：PS 模式、危险命令首词、
    高危 API、内嵌 shell 命令、重定向越界。命中返回拦截说明，安全返回 None。
    """
    if not body or not body.strip():
        return None
    for pat, desc in _PS_PATTERNS:
        if pat.search(body):
            return f"命令拦截：{desc}"
    for h in _command_heads(body):
        if h in _ALL_DANGEROUS_SET:
            return f"命令拦截：检测到危险命令 `{h}`"
    for pat, why in _CODE_HIGH_RISK:
        if re.search(pat, body):
            return f"命令拦截：内联代码调用了 `{why}`（不可逆或绕过黑名单）"
    for m in _EMBEDDED_Q.finditer(body):
        inner = m.group(1) or m.group(2) or ""
        for h in _command_heads(inner):
            if h in _ALL_DANGEROUS_SET:
                return f"命令拦截：内联代码里嵌了危险命令 `{h}`"
    danger = _check_redirect_targets(body)
    if danger:
        return danger
    return None


def _is_encoded_cmd_flag(tok: str) -> bool:
    """判断 powershell 参数是否为 -EncodedCommand 或其合法缩写。

    powershell.exe 允许参数前缀缩写（-e / -enc / -ec / -encodedcommand …），
    只匹配全称会让 `-e <base64>` 绕过检查。这里用"是否为 encodedcommand 的前缀"
    精确判定：-executionpolicy（-ex…）不匹配，不会误伤。
    """
    t = tok.lower().lstrip("-")
    return bool(t) and ("encodedcommand".startswith(t) or t == "ec")


def _check_command_safety(command: str) -> str | None:
    """前置安全检查。命中黑名单返回危险描述字符串，安全返回 None。

    拦截策略（按执行顺序）：
    1) 注入模式：命令拼接 `cmd1 && cmd2` 等，无论啥 shell 一律拦截
    2) 识别 shell 类型（cmd / powershell / 其它），分别做精确匹配；
       cmd 侧按"命令首词"匹配，参数位出现的同名词（ruff format .）不拦
    """
    stripped = command.strip()
    if not stripped:
        return None

    # ---- 1. 注入模式扫描 ----
    # 先屏蔽引号内的字面量：`echo "a|b|c"` 的管道不是命令拼接，不该拦。屏蔽后
    # `cmd1 && cmd2`、`a | b` 这类真实拼接照拦；命中片段从原串取（等长替换保证位置一致）。
    scan = _mask_quoted(stripped)
    for pat in _INJECTION_PATTERNS:
        m = pat.search(scan)
        if m:
            snippet = stripped[m.start():m.end()]
            return f"命令拦截：检测到 shell 拼接模式 `{snippet}`，存在命令注入风险"

    # ---- 1.5 重定向目标越界检查 ----
    # cwd 只限制命令本身的工作目录，重定向目标不受 cwd 约束，
    # 必须单独拦截（echo x > ..\..\etc\passwd 会写出项目根）。
    redirect_danger = _check_redirect_targets(stripped)
    if redirect_danger:
        return redirect_danger

    # ---- 2. 危险命令匹配 ----
    heads = _command_heads(stripped)
    head = heads[0] if heads else ""

    if head in ("powershell", "pwsh"):
        # PowerShell 模式：跳过 powershell.exe [-nop] [-command] 包裹，检查 -Command 后的正文
        try:
            tokens = shlex.split(stripped, posix=False)
        except ValueError:
            tokens = [stripped]
        body_start = 0
        for i, t in enumerate(tokens):
            if _is_encoded_cmd_flag(t):
                return "命令拦截：PowerShell -EncodedCommand 可能掩盖危险操作，拒绝执行"
            if t.lower() in ("-command", "-c", "-cmd"):
                body_start = i + 1
                break
        body = " ".join(tokens[body_start:])
        # 正文统一扫描（#5）：PS 危险模式 + cmd 风格首词 + 高危 API + 内嵌命令 + 重定向
        danger = _scan_code_body(body)
        if danger:
            return danger
    else:
        # cmd 或裸命令：按每段的命令首词判定（路径/包装/后缀已归一）
        for h in heads:
            if h in _CMD_DANGEROUS_SET:
                return f"命令拦截：检测到危险命令 `{h}`"
        # #5：python -c / node -e / perl -e 等"解释器 + 内联代码"绕过点——
        # 首词是解释器、破坏藏在正文里，这里对正文再扫一遍。
        _interp, code = _extract_inline_code(_split_tokens(stripped))
        if code:
            danger = _scan_code_body(code)
            if danger:
                return danger

    return None


def _try_split_args(command: str) -> list[str] | None:
    """尝试把命令拆分成参数列表（shell=False 模式）。成功返回列表，失败返回 None。

    Windows 上有管道/重定向/拼接符时无法用参数模式执行，此时返回 None 退回 shell=True。
    """
    if _SHELL_ONLY_SIGNS.search(command):
        return None  # 有 shell 专有语法，只能用 shell=True
    try:
        args = shlex.split(command, posix=False)
        return [_strip_wrapping_quotes(a) for a in args] if args else None
    except ValueError:
        return None
