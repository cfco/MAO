"""Shell 安全的共享规则层：token 拆分、重定向越界、密钥文件名、PS 开关判定、子进程辅助。

为什么独立成模块（2026-11-04，自 shell_safety.py 拆出守「单文件 ≤500 行」约定）：
这些都是"纯函数规则"（给定字符串 → 判定/归一结果，不依赖黑名单状态），被
shell_safety（编排层：注入/黑名单/内联扫描）与 builtin（工具层：密钥文件口径）
共同消费。shell_safety 保留 re-export，既有导入面逐字不变。
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess

from ..core.constants import SECRET_NAME_SUBSTRINGS, SECRET_NAME_TAILS

# Windows 可执行后缀：format.com / del.exe 等真实二进制要归一到命令词再比对
_EXE_SUFFIXES = (".exe", ".com")
# 命令分隔符：一次请求里可能串多条命令（echo x & del /s y），每段的首词都要查。
# 必须含换行：cmd.exe 也把换行当命令分隔符执行，而含 `>` 的命令一律走 shell=True，
# 于是 `echo a>nul` + 换行 + `del /f /q x` 的首词只会是 echo，整条黑名单被绕过（实测放行）。
_CMD_SEPARATOR = re.compile(r"&&|\|\||[&|;\r\n]")


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


# shell 重定向操作符：> >> < 2> 2>> 1> &> >| <> 等，后跟目标文件名。
# 用于检测「echo x > ..\..\etc\passwd」这类通过重定向绕过 cwd 限制的写越界。
# 目标捕获：引号包裹的串 或 不含空白/特殊符号的 token。
# `>|`（POSIX clobber 强制覆盖）必须单列：旧写法的操作符表只有 >> > <，而目标字符类
# 排除了 `|`，于是 `echo x >| /etc/evil` 整条 match 失败、越界检查一次都不进（实测放行）。

_REDIRECT_RE = re.compile(
    r"(?<![<>=!])"  # 排除 => / >= / -> 这类比较与箭头运算符粘在操作符前的写法
    r"(?:\d*&?)?"  # 可选 fd 前缀：1/2/&（1>, 2>>, &>）
    r"(?:>>|>\||<>|>|<)"  # 操作符：长操作符优先匹配（>> 先于 >、>| 先于 >）
    r"\s*"
    r"(\"[^\"]*\"|'[^']*'|[^\s&|<>]+)"  # 目标：引号串 或 非特殊 token
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
        # 环境变量/家目录写法（%USERPROFILE%\\x、$env:TEMP\\x、~/x）要展开后才决定落点，
        # 静态检查看不到展开结果 → 一律当作可能写出项目根拒绝（实测旧写法直接放行）。
        if "%" in target or target.startswith(("$", "~")):
            return (
                f"命令拦截：重定向目标 `{target}` 含环境变量/家目录展开，落点无法静态确认，已拒绝"
            )
        # 设备名放行（NUL/CON/PRN/AUX/COMx/LPTx 等）
        if re.match(r"^(nul|con|prn|aux|com[1-9]|lpt[1-9])(\.|$)", low):
            continue
        # 绝对路径（盘符 C: 或根斜杠）直接越界
        if re.match(r"^[a-zA-Z]:", target) or target.startswith(("/", "\\")):
            return f"命令拦截：重定向目标 `{target}` 是绝对路径，可能写出项目根目录，已拒绝"
        # 相对路径穿越：含 .. 的目标可能逃出项目根
        if ".." in target.replace("\\", "/").split("/"):
            return f"命令拦截：重定向目标 `{target}` 含路径穿越(..)，可能写出项目根目录，已拒绝"
    return None


def _norm_head(raw: str) -> str:
    """把 token 归一为可比较的命令词：剥包裹引号、cmd 的 `^` 转义与 `%VAR%` 两侧百分号，
    取 basename、去 exe 后缀。

    归一不做全的话，`"del" /s x`、`C:\\...\\format.com`、`de^l /f x`、`%COMSPEC% /c del x`
    这类写法各自逃过一张词表里的某一条。
    """
    low = _strip_wrapping_quotes(raw).replace("^", "")
    low = low.replace("\\", "/").rsplit("/", 1)[-1].strip("%").lower()
    for suf in _EXE_SUFFIXES:
        if low.endswith(suf):
            return low[: -len(suf)]
    return low


def _head_word(segment: str) -> str:
    """取一段命令中"实际被执行"的命令首词（小写）。

    归一化见 _norm_head（剥引号、去 cmd 的 `^` 转义与 `%…%`、取 basename、去 .exe/.com 后缀）。
    `cmd /c` 这类包装不在这里剥：它的首词就是 cmd 本身，被包装的正文由 _command_heads
    递归扫描（见 _wrapped_payload 的四个绕过点）。
    """
    s = segment.strip()
    if not s:
        return ""
    tokens = _split_tokens(s) or [s]  # 拆词统一走 _split_tokens：ValueError 已兜成空列表
    if not tokens:
        return ""
    # 剥引号后可能仍带空格（整个命令被一对引号包住），取第一个词才是命令词
    parts = _strip_wrapping_quotes(tokens[0]).strip().split()
    return _norm_head(parts[0]) if parts else ""


def _split_tokens(command: str) -> list[str]:
    """把命令拆成参数列表；**全模块唯一的 shlex 入口**，失败（引号不闭合等 ValueError）返回空列表。

    原先 _command_head / _try_split_args 等各自再包一层 shlex.split + except，
    容错口径散在多处、改一处漏一处；统一收在这里。
    """
    try:
        return shlex.split(command, posix=False)
    except ValueError:
        return []


_UNQUOTE_CHARS = re.compile(r'["\']')

# ==============================================================
# 密钥文件访问拦截：read_file 有守卫，但 `type .env` / `cat .env` 走 run_shell
# 就能整份读出（首词既不在 cmd 也不在 POSIX 危险词表里），而工具输出是直接外发给
# 外部主的。文件名口径与 builtin._is_secret_file 共用 _is_secret_basename。
# ==============================================================


def _is_secret_basename(name: str) -> bool:
    """该文件名是否属于"工具不得外发内容"的密钥文件（模板 .env.example 除外）。

    只认**开头**是 secret/credential 的名字：用 `in` 判定会把 `NODE_A_KEY=super…` 这类
    参数值一并误拦（实测：`python -c "print('...supersecretvalue...')"` 整条命令被拒）。
    名字里带 `.example` 的一律放行：模板及其备份（`.env.example.bak`）不含凭据，
    而文档正是教人从模板复制。
    """
    n = name.strip().lower().lstrip("<>;&|")
    if not n or ".example" in n:
        return False
    return n == ".env" or n.startswith(".env.") or n.startswith(("secret", "credential"))


# `.env` 及其派生名（`.env.local`），前后不能粘字母数字（避开 `myenv`、`.environment`、
# 以及可读写、也可安全外发的模板 `.env.example`）；`%`/`$`/`{`/`}` 也要排除，否则
# Windows 的 `%comspec%` 这类变量名尾部会被当成 `.env` 误拦。
_DOTENV_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_%${])\.env(\.[A-Za-z0-9_-]+)?(?![A-Za-z0-9])", re.IGNORECASE
)


def _check_secret_file_access(command: str) -> str | None:
    """命令正文里出现密钥文件名一律拒绝（读写方向都拦：写入等于把 key 换到外部端点）。

    两条判定合成一处：token 走 basename 比对（`cat /etc/paid-secrets.json`），整串正则
    走 dotenv 形态（`echo x>.env` 这种重定向不留空格时 shlex 给出 `x>.env`，basename
    比对看不见）。`.env` 就躺在项目根，能读它的命令不需要越界，故按名字一律拦。
    """
    for raw in _split_tokens(command):
        name = _strip_wrapping_quotes(raw).replace("\\", "/").rsplit("/", 1)[-1]
        if _is_secret_basename(name):
            return f"命令拦截：`{name}` 是密钥文件，工具层不通过 shell 外发其内容，已拒绝"
    for m in _DOTENV_TOKEN_RE.finditer(_strip_wrapping_quotes(command)):
        # 命中形态还要过一遍名字判定：`.env.example`（模板）也会被这条正则扫到，
        # 但它不含凭据、且是文档里教人复制的来源，不能拦。
        if _is_secret_basename(m.group(0)):
            return f"命令拦截：`{m.group(0)}` 是密钥文件，工具层不通过 shell 外发其内容，已拒绝"
    return None


def _check_powershell_encoding(command: str) -> str | None:
    """整条命令（含被引号包裹的正文）里出现 powershell + -EncodedCommand 一律拒。

    为什么不能只按首词判定：`echo hi & powershell -enc AAAA=` 的首词是 echo，
    `cmd /c "powershell -enc AAAA="` 的正文藏在引号里（shlex 拆出来是一整串）——
    两种写法都绕过原先"首词是 powershell 才检查"的分支。base64 编码的命令正文静态扫不动，
    只能整类拒绝。`-executionpolicy` 不会被误伤：前缀判定只认 encodedcommand 的缩写。
    """
    flat = _UNQUOTE_CHARS.sub(" ", command)
    for seg in _CMD_SEPARATOR.split(flat):
        toks = [t.lower() for t in seg.split()]
        if not any(t in ("powershell", "powershell.exe", "pwsh", "pwsh.exe") for t in toks):
            continue
        for t in toks:
            if t.startswith("-") and _is_encoded_cmd_flag(t):
                return "命令拦截：PowerShell -EncodedCommand 可能掩盖危险操作，拒绝执行"
    return None


def _is_encoded_cmd_flag(tok: str) -> bool:
    """判断 powershell 参数是否为 -EncodedCommand 或其合法缩写。

    powershell.exe 允许参数前缀缩写（-e / -enc / -ec / -encodedcommand …），
    只匹配全称会让 `-e <base64>` 绕过检查。这里用"是否为 encodedcommand 的前缀"
    精确判定：-executionpolicy（-ex…）不匹配，不会误伤。
    """
    t = tok.lower().lstrip("-")
    return bool(t) and ("encodedcommand".startswith(t) or t == "ec")


def _runner_targets(segment: str) -> list[str]:
    """xargs / find 段里"实际会被执行"的命令词（Runner 自己的参数不算）。

    - xargs：第一个非选项参数即要执行的命令（`-I{} rm` 的 rm 在占位符之后，
      取第一个非选项词即可；多取一个选项词只是多查一个无害词，不漏判就行）；
    - find：-exec/-execdir 的后一个词；-delete 等价递归删除，按 rm 同等拦截。
    """
    tokens = [_strip_wrapping_quotes(t) for t in _split_tokens(segment)]
    if not tokens:
        return []
    head = tokens[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    if head == "xargs":
        for t in tokens[1:]:
            if t.startswith("-"):
                continue
            return [t.replace("\\", "/").rsplit("/", 1)[-1].lower()]
        return []
    if head == "find":
        out: list[str] = []
        for i, t in enumerate(tokens[1:], start=1):
            low = t.lower()
            if low == "-delete":
                out.append("rm")  # -delete 就是递归删除，按 rm 拦
            elif low in ("-exec", "-execdir") and i + 1 < len(tokens):
                out.append(tokens[i + 1].replace("\\", "/").rsplit("/", 1)[-1].lower())
        return out
    return []


# ---------- 子进程辅助（2026-11-04 自 builtin.py 拆出；builtin 保留 re-export） ----------


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """把超时子进程连同其子孙一起杀掉（跨平台）。

    为什么不能只 proc.kill()：subprocess 的 timeout 只终止**直接**子进程，
    `sh -c "sleep 60 &"` / `nohup ... &` / `setsid ...` 的孙进程会残留成孤儿继续跑
    （Linux 上这类写法是常规操作，实测一条超时命令能留下活过数分钟的孙进程）。
    - POSIX：spawn 时带 start_new_session=True，子进程自成进程组且 pgid==pid，
      直接 killpg(proc.pid) 全组收。用 pid 而非 getpgid(pid)：直接子进程可能已被
      回收，但只要组内还有子孙，进程组就仍在，killpg 依然有效；
    - Windows：taskkill /F /T /PID 按进程树杀（不依赖进程组）。
    两条路都失败（进程组已不存在等）就退化为 proc.kill()——Popen 内部对已退出
    进程会短路，不会误杀 pid 回收后同号的新进程。
    """
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=15,
                check=False,
            )
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        try:
            proc.kill()
        except OSError:
            pass


# 明确拒绝读写的密钥文件名口径在 shell_safety._is_secret_basename（`.env` 与其派生、
# 名字含 secret/credential 的文件；模板 `.env.example` 仍可读写），文件与 shell 两条
# 通道共用一份，免得加一类名字时漏改一侧。


def _child_env() -> dict[str, str]:
    """给子进程一份**剔掉密钥形态变量**的环境副本。

    为什么需要：`config._load_dotenv` 把 `.env` 的全部值注入 `os.environ`（含
    `NODE_*_KEY` / `*_API_TOKEN`），而 `run_shell` 会把命令 stdout 原样带回给外部主 /
    任意 MCP 宿主，且工具输出根本不过 `redact.py` —— 一条 `set` 就能把这台机器的全部
    API key 打包带走（实测子进程确实能看到 `NODE_A_KEY`）。命令黑名单声明过"不是沙箱"，
    但"能执行命令"和"能顺手抄走凭据"是两回事。
    只按名字判断（词表见 core/constants 的 SECRET_NAME_TAILS / SECRET_NAME_SUBSTRINGS，
    与 redact 脱敏共用一份）：三种命名都要拦——
    - 分段式（`NODE_A_KEY` → 末段 `KEY`）；
    - 紧凑式（`MYKEY` / `GHTOKEN` / `OPENAIKEY` → 整名以密钥词结尾）；
    - 中段式（`AWS_ACCESS_KEY_ID` / `AUTH_TOKEN_V2` / `CREDENTIALS` → 密钥词在名字
      中段、末段是 ID/V2/S 这类后缀）。2026-11-04 审计实测第三类整类漏掉，
      一条 `set`/`env` 就能把值原样带进工具输出，故补子串级判定。
    这里刻意**宁严勿漏**：误剔一个同形词（如 `MONKEY`）只是子进程少一个无关变量，
    漏掉一个密钥则是把整机凭据交给工具输出（展示侧脱敏现已同口径，见 redact.py）。
    """
    out: dict[str, str] = {}
    for k, v in os.environ.items():
        name = k.upper()
        if name.rsplit("_", 1)[-1] in SECRET_NAME_TAILS or any(
            word in name for word in SECRET_NAME_SUBSTRINGS
        ):
            continue
        out[k] = v
    return out


def _decode(b: bytes) -> str:
    """命令输出解码：优先 utf-8，失败退 gbk（Windows 中文环境），再不行丢字符。"""
    for enc in ("utf-8", "gbk"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")
