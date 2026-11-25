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

from .shell_rules import (  # noqa: F401 - re-export：历史导入面（builtin/测试）不变
    _CMD_SEPARATOR,
    _DOTENV_TOKEN_RE,
    _EXE_SUFFIXES,
    _QUOTED_RE,
    _REDIRECT_RE,
    _UNQUOTE_CHARS,
    _check_powershell_encoding,
    _check_redirect_targets,
    _check_secret_file_access,
    _head_word,
    _is_encoded_cmd_flag,
    _is_secret_basename,
    _mask_quoted,
    _norm_head,
    _runner_targets,
    _split_tokens,
    _strip_wrapping_quotes,
)

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
    "format",  # 格式化磁盘
    "fdisk",  # 分区（旧版 Windows）
    "diskpart",  # 磁盘分区工具（可删除卷）
    "clean",  # diskpart clean 清磁盘
    "chkdsk",  # 修复磁盘（可能强制卸载卷）
    "rd",  # rd /s /q 递归静默删除目录（比 rm -rf 还狠）
    "rmdir",  # 同上
    "del",  # del /f /q /s 强制静默递归删除文件
    "erase",  # 同上
    "shutdown",  # 关机/重启
    "reboot",  # 重启（旧命令）
    "reg",  # reg delete / reg import 删改注册表
    "schtasks",  # schtasks /delete 删除计划任务
    "taskkill",  # taskkill /f /im 强制杀进程
    "wmic",  # wmic process delete 删除进程
    "bootcfg",  # 修改引导配置
    "bcdedit",  # 修改引导配置（Win Vista+）
    "syskey",  # 系统数据库加密（不可逆）
]
_CMD_DANGEROUS_SET = frozenset(_CMD_DANGEROUS)

# 命令 Runner 的"正文开关"：这些包装把真正的命令藏在参数里执行，只归一外层命令词
# 等于放行内层（`cmd /c "del /f /q x"` 的首词变成 cmd）。cmd 用斜杠开关，shell 家族
# 用 `-c`（含 `-lc` 这类短选项簇，见 _SHELL_CODE_FLAG）。
_RUNNER_HEADS = {"cmd", "comspec", "sh", "bash", "zsh", "dash", "ksh"}
_TRANSPARENT_HEADS = {"call", "command"}
_RUNNER_FLAGS = {
    "cmd": frozenset({"/c", "/k"}),
    "shell": frozenset({"-c"}),
}
# Runner 展开的总量预算（2026-11-04 审计，见 _unwrap_runners）：正常命令的嵌套不超过
# 两三层，离预算差几个数量级；只有刻意构造的超深包装才会触顶，触顶即整条拒绝（宁可误拦）。
_RUNNER_SCAN_CHAR_BUDGET = 65536
_RUNNER_MAX_TEXTS = 64

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
    re.compile(r"(?:&&|\|\|)\s*\S+"),  # cmd1 && cmd2 / cmd1 || cmd2
    re.compile(r"[|&]\s*\S+\s*[|&]"),  # 中间嵌套管道
    re.compile(
        r";\s*\S+\s*;\s*\S+"
    ),  # 分号多段（Unix shell 风格，Windows cmd 不识别但 PowerShell 可）
    re.compile(r"\$\([^)]+\)"),  # 命令替换 $(...)
    re.compile(r"`[^nrt]", re.DOTALL),  # PowerShell 反引号转义接非控制字符
]

# shell=False 无法支持的 shell 专有语法特征（检测到则退回 shell=True）
_SHELL_ONLY_SIGNS = re.compile(r"[|><&]|\|\||&&")

# 成对引号包裹的内容（单/双引号）
# `<>`（读写打开）同理补上。


def _wrapped_payload(tokens: list[str]) -> str:
    """取 `cmd /c <正文>` / `sh -c <正文>` 这类包装里的正文；不是包装则返回空串。

    为什么必须单独取：包装的外层词（cmd/sh）不在危险词表里，旧写法只剥 `cmd /c` 的裸
    拼写，于是引号（`"cmd" /c "del x"`）、全路径（`C:\\Windows\\System32\\cmd.exe /c del x`）、
    `%COMSPEC%`、POSIX 的 `sh -c "rm -rf x"` 都把内层破坏命令留在原地不被扫描——整条黑名单
    形同虚设。这里按归一后的命令词判定，再按开关（cmd 用 /c /k、shell 用 -c / -lc）取正文。
    """
    if not tokens:
        return ""
    head = _norm_head(_strip_wrapping_quotes(tokens[0]))
    flags = (
        _RUNNER_FLAGS["cmd"]
        if head in ("cmd", "comspec")
        else _RUNNER_FLAGS["shell"]
        if head in _RUNNER_HEADS
        else None
    )
    if flags is None:
        return ""
    for i, tok in enumerate(tokens[1:], start=1):
        flag = _strip_wrapping_quotes(tok).strip().lower()
        if flag in flags or (head in _RUNNER_HEADS and _SHELL_CODE_FLAG.fullmatch(flag)):
            return " ".join(_strip_wrapping_quotes(t) for t in tokens[i + 1 :])
        if flag.startswith("-") or flag.startswith("/"):
            continue
        return ""  # 开关还没出现就来了位置参数（`cmd /d something`）：没有正文可扫
    return ""


def _unwrap_runners(command: str) -> list[str] | None:
    """把命令连同**每一层 Runner 包装的正文**一起展开（迭代、带总量预算）。

    Runner = `cmd /c`、`cmd.exe /k`、`%COMSPEC% /c`、`sh/bash/zsh/dash/ksh -c`。它们的
    破坏动作都藏在正文里，展开一次就等于给内层再走一遍全部扫描（危险词、内联代码、
    重定向越界）。迭代而非"只剥一层"的理由见 _wrapped_payload。

    为什么改为迭代 + 预算（2026-11-04 审计）：旧实现递归且深度硬限 4 层，第 5 层起的
    正文不再展开，`cmd /c ` ×5 + `del /f /q x` 的 del 永远进不了 heads——深度上限
    成了可复现的绕过点。但完全不设限也不行：极端嵌套会做 O(n²) 次分词。正文每剥一层
    严格变短（丢掉 runner 头与开关），递归天然会终止，所以用「展开文本数 + 累计字符
    预算」代替层数上限；预算耗尽返回 None，调用方**整条拒绝**（宁可误拦也不放行），
    正常命令离预算差几个数量级，不受影响。
    """
    out = [command]
    seen_chars = len(command)
    queue = [command]
    while queue:
        text = queue.pop(0)
        for seg in _CMD_SEPARATOR.split(text):
            head = _head_word(seg)
            if head in _TRANSPARENT_HEADS:
                tokens = _split_tokens(seg.strip())
                if len(tokens) > 1:
                    payload = " ".join(_strip_wrapping_quotes(t) for t in tokens[1:])
                    out.append(payload)
                    seen_chars += len(payload)
                    if seen_chars > _RUNNER_SCAN_CHAR_BUDGET or len(out) > _RUNNER_MAX_TEXTS:
                        return None
                    queue.append(payload)
                continue
            if head not in _RUNNER_HEADS:
                continue
            # cmd 执行前会还原 `^` 转义，正文里留着转义符就扫不到真实命令词
            payload = _wrapped_payload(_split_tokens(seg.strip())).replace("^", "")
            if not payload:
                continue
            out.append(payload)
            seen_chars += len(payload)
            if seen_chars > _RUNNER_SCAN_CHAR_BUDGET or len(out) > _RUNNER_MAX_TEXTS:
                return None
            queue.append(payload)
    return out


def _command_heads(command: str) -> list[str] | None:
    """把命令按 shell 分隔符切段，返回每段"实际被执行"的命令词（全部要查）。

    Runner 展开超预算时返回 None：调用方必须整条拒绝（防绕过，见 _unwrap_runners）。

    只查整条命令的首词会漏掉串联在后面的危险命令：`echo x & del /s y` 的
    首词是 echo，但真正执行删除的是 del。`&&`/`||` 已被注入模式拦下，
    这里的切段是单 `&`、`|`、`;` 串联场景的兜底。

    每段除首词外还收两类"真正会被执行、但不在首词位置"的词，漏掉等于顶层放行
    （实测 2026-10-22：`ls | xargs rm -f` 三层防护一层都不触发）：
    - xargs / find 的目标命令（_runner_targets）；
    - Runner 包装正文里的命令词（_unwrap_runners 展开后各自再切段一次）。
    """
    texts = _unwrap_runners(command)
    if texts is None:
        return None
    words: list[str] = []
    for text in texts:
        for seg in _CMD_SEPARATOR.split(text):
            head = _head_word(seg)
            if head:
                words.append(head)
            words.extend(_runner_targets(seg))
    return words


_PS_PATTERNS = [(re.compile(p, re.IGNORECASE), desc) for p, desc in _PS_DANGEROUS]


# ==============================================================
# 解释器内联代码防护（#5）：python -c / node -e / perl -e … 首词是解释器、
# 破坏逻辑藏在代码正文里，只扫"命令首词"会被整体绕过。这里对"内联代码正文"
# 再扫一遍危险 API / 内嵌 shell 命令 / 重定向越界。纵深防御、非沙箱。
# ==============================================================
_INTERP_INLINE = {
    "python",
    "python3",
    "python2",
    "py",
    "pypy",
    "pypy3",
    "node",
    "nodejs",
    "deno",
    "bun",
    "ruby",
    "perl",
    "php",
    "osascript",
    "tclsh",
    "wish",
    "expect",
    # POSIX shell 家族同属"首词是解释器、破坏藏在正文里"的绕过点。Windows 上装了
    # Git-Bash/WSL 就能执行，而危险词表是 Windows 向的（rm 只在正文扫描里生效），
    # 不收进本集合则 `bash -c "rm -rf ~"` 三层防护一层都不触发。
    "sh",
    "bash",
    "zsh",
    "dash",
    "ksh",
}
# 携带"内联代码字符串"的开关（powershell/pwsh 的 -Command 走它自己的分支）
_INLINE_CODE_FLAGS = {"-c", "-e", "--eval", "-r", "--rc", "-import", "--import"}
# POSIX shell 的单横线短选项簇：`bash -lc` / `sh -xlc` 的正文位置与 -c 完全一致。
# 只认单横线，`--norc`（双横线、也以 c 结尾）不会被误当成内联开关。
_SHELL_CODE_FLAG = re.compile(r"-[a-z]{0,3}c\Z")
_POSIX_SHELL_INTERP = {"sh", "bash", "zsh", "dash", "ksh"}
# POSIX 侧破坏性命令（内联代码常调 sh/os.system，补齐 Windows 集合看不到的词）。
# 顶层判定同样用这份集合（见 _check_command_safety）：Linux 上 rm/mkfs/dd 是原生命令，
# 历史缺口是"只在内联正文里拦、顶层放行"——实测 2026-10-22：顶层 `rm -rf /` 直过，
# 与 run_shell 描述声明的"rm/del/format 都拦"契约相悖（CI 跑 ubuntu 却没人踩到，
# 因为测试只覆盖了内联场景）。
_POSIX_DANGEROUS_SET = {
    "rm",
    "rmdir",
    "mkfs",
    "dd",
    "shred",
    "mkfstools",
    "truncate",
    "mkswap",
    "sfdisk",
    "parted",
    "wipefs",
    "poweroff",
    "halt",
}
# 带工具变体后缀的首词（mkfs.ext4 / mkfs.xfs / mkfs.vfat…）：精确匹配拦不住，按前缀兜。
# 只放 mkfs——`mk` 开头的 mkdir/mkfifo 是正常命令，绝不能按前缀误伤。
_POSIX_DANGEROUS_PREFIXES = ("mkfs",)
_ALL_DANGEROUS_SET = _CMD_DANGEROUS_SET | _POSIX_DANGEROUS_SET


def _is_dangerous_word(word: str) -> bool:
    """该命令词是否破坏性。顶层与内联正文共用同一判定，避免两处口径漂移。"""
    if word in _ALL_DANGEROUS_SET:
        return True
    return any(word.startswith(p) for p in _POSIX_DANGEROUS_PREFIXES)


# 内联代码里的"高危 API"：无 shell 对应词、破坏性/绕过黑名单意图明显
_CODE_HIGH_RISK = [
    (r"\bshutil\s*\.\s*rmtree\b", "shutil.rmtree（递归删除）"),
    (
        r"\bos\s*\.\s*(remove|unlink|removedirs|kill|killpg|startfile|ftruncate)\b",
        "os 文件/进程删除",
    ),
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

# 内联代码里"读文件再编码外带"的形态（2026-11-04 审计）：`open('.en'+'v')` 靠字符串
# 拼接绕过 .env 文件名识别，再 base64 让输出掩码失效——三层凭据防护被一条命令穿透。
# 只在代码正文里查（不查普通 shell 命令）：正常开发代码极少在 open() 参数里拼字符串。
_CODE_SECRET_READ = [
    # open() / readFileSync 参数里出现"字符串字面量 + 拼接"：拼路径读密钥文件的典型写法
    (
        re.compile(r"\bopen\s*\([^)]*['\"][^'\"]*['\"]\s*\+"),
        "open() 参数含字符串拼接（疑似拼出密钥文件名）",
    ),
    (
        re.compile(r"\bopen\s*\([^)]*\+\s*['\"]"),
        "open() 参数含字符串拼接（疑似拼出密钥文件名）",
    ),
    (
        re.compile(r"\breadFile(?:Sync)?\s*\([^)]*\+"),
        "Node readFile 参数含字符串拼接",
    ),
    # 短字面量拼接出 ".env" 的碎片形态：'.en'+'v'、'.'+'env'、".en" +"v"
    (
        re.compile(r"['\"]\.en['\"]|['\"]\.['\"]\s*\+\s*['\"]env|['\"]env['\"]\s*\+"),
        "字符串字面量疑似 .env 的拆分拼接",
    ),
]


def _extract_inline_code(tokens: list[str]) -> tuple[str | None, str]:
    """若首 token 是内联解释器且带内联代码开关，返回 (解释器名, 代码正文)；否则 (None, "")。"""
    if not tokens:
        return None, ""
    interp = tokens[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    if interp.endswith(".exe"):
        interp = interp[:-4]
    if interp not in _INTERP_INLINE:
        return None, ""
    is_shell = interp in _POSIX_SHELL_INTERP
    for i, t in enumerate(tokens[1:], start=1):
        flag = _strip_wrapping_quotes(t).strip().lower()
        hit = flag in _INLINE_CODE_FLAGS or (is_shell and _SHELL_CODE_FLAG.fullmatch(flag))
        if hit and i + 1 < len(tokens):
            return interp, _strip_wrapping_quotes(tokens[i + 1])
    return None, ""  # 是解释器但无内联代码（如 python script.py：正文不可静态扫描，放行）


def _scan_code_body(body: str) -> str | None:
    """扫描"代码正文 / 内联脚本"里的破坏性内容：PS 模式、危险命令首词、
    高危 API、密钥文件拼接读取、内嵌 shell 命令、重定向越界。
    命中返回拦截说明，安全返回 None。
    """
    if not body or not body.strip():
        return None
    for pat, desc in _PS_PATTERNS:
        if pat.search(body):
            return f"命令拦截：{desc}"
    heads = _command_heads(body)
    if heads is None:
        return "命令拦截：Runner 包装嵌套过深，已拒绝（防绕过黑名单）"
    for h in heads:
        if _is_dangerous_word(h):
            return f"命令拦截：检测到危险命令 `{h}`"
    for pat, why in _CODE_HIGH_RISK:
        if re.search(pat, body):
            return f"命令拦截：内联代码调用了 `{why}`（不可逆或绕过黑名单）"
    for pat, why in _CODE_SECRET_READ:
        if pat.search(body):
            return f"命令拦截：内联代码疑似拼接读取密钥文件（{why}），已拒绝"
    # 编码外带兜底：同一正文里既 open() 读文件又用 base64/codecs 编码，读什么已不重要
    # ——这是"读出来再编码绕过输出掩码"的固定组合，宁严勿漏。
    if re.search(r"\bopen\s*\(", body) and re.search(
        r"\bbase64\b|\bcodecs\b|\bbinascii\b|b64encode|b64encodeSync", body
    ):
        return "命令拦截：内联代码读文件后做 base64/codecs 编码（疑似绕过输出掩码外带），已拒绝"
    for m in _EMBEDDED_Q.finditer(body):
        inner = m.group(1) or m.group(2) or ""
        inner_heads = _command_heads(inner)
        if inner_heads is None:
            return "命令拦截：Runner 包装嵌套过深，已拒绝（防绕过黑名单）"
        for h in inner_heads:
            if _is_dangerous_word(h):
                return f"命令拦截：内联代码里嵌了危险命令 `{h}`"
    danger = _check_redirect_targets(body)
    if danger:
        return danger
    return None


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
    if "\r" in command or "\n" in command:
        # 与上面的切段是双保险：一条工具调用就该是一条命令，直接拦在入口，
        # 免得后续任何"按整串处理"的环节把第二行当参数放过。
        return "命令拦截：一条命令里不能含换行（cmd 会把换行当分隔符执行后面的内容），已拒绝"

    # ---- 1. 注入模式扫描 ----
    # 先屏蔽引号内的字面量：`echo "a|b|c"` 的管道不是命令拼接，不该拦。屏蔽后
    # `cmd1 && cmd2`、`a | b` 这类真实拼接照拦；命中片段从原串取（等长替换保证位置一致）。
    scan = _mask_quoted(stripped)
    for pat in _INJECTION_PATTERNS:
        m = pat.search(scan)
        if m:
            snippet = stripped[m.start() : m.end()]
            return f"命令拦截：检测到 shell 拼接模式 `{snippet}`，存在命令注入风险"

    # ---- 1.2 密钥文件访问：`type .env` / `cat .env` / `echo k>>.env` 一律拒 ----
    # read_file 早就拦了 .env，但 shell 侧的内置命令（type/cat/more）首词都不在两张
    # 危险词表里，一条命令就能把全部 key 读进 stdout 外发给外部主。
    secret_danger = _check_secret_file_access(stripped)
    if secret_danger:
        return secret_danger

    # ---- 1.5 重定向目标越界检查 ----
    # cwd 只限制命令本身的工作目录，重定向目标不受 cwd 约束，
    # 必须单独拦截（echo x > ..\..\etc\passwd 会写出项目根）。
    redirect_danger = _check_redirect_targets(stripped)
    if redirect_danger:
        return redirect_danger

    # ---- 1.6 PowerShell -EncodedCommand：按内容查、不按首词查（见函数文档的两个绕过点）----
    ps_enc = _check_powershell_encoding(stripped)
    if ps_enc:
        return ps_enc

    # ---- 2. 危险命令匹配（无条件、对全部 heads）----
    # 2026-11-04 审计 P0：旧写法按"heads 里出现 powershell 就走 PS 分支"分流，
    # else 分支里的首词黑名单与内联扫描被整体跳过——`del /f /q x & powershell -c "1"`
    # 的 del 排在 PS 段之前，既不在 body 里、也不再被检查，黑名单形同虚设（实测放行）。
    # 修法：首词黑名单与内联代码扫描对**所有** heads 无条件执行；PS 分支只保留
    # "额外扫 -Command 正文"这一层增量防护。
    heads = _command_heads(stripped)
    if heads is None:
        return "命令拦截：Runner 包装嵌套过深，已拒绝（防绕过黑名单）"
    # 用 _ALL_DANGEROUS_SET（Windows + POSIX 合并）而非只看 Windows 词表：
    # Linux 上 rm/mkfs/dd 是原生命令，顶层放行与工具描述的拦截声明相悖
    # （2026-10-22 实测顶层 `rm -rf /` 直过；两边词表合并后 Windows 侧
    # 只是多认几个 Git-Bash/WSL 里可能出现的词，正常开发命令不受影响）。
    for h in heads:
        if _is_dangerous_word(h):
            return f"命令拦截：检测到危险命令 `{h}`"

    # ---- 3. 解释器内联代码扫描（无条件）----
    # #5：python -c / node -e / perl -e 等"解释器 + 内联代码"绕过点——
    # 首词是解释器、破坏藏在正文里，这里对正文再扫一遍。逐层展开 Runner 包装，
    # 否则 `cmd /c "python -c …"` 的外层词是 cmd，内联正文永远进不了这条扫描。
    texts = _unwrap_runners(stripped)
    if texts is None:
        return "命令拦截：Runner 包装嵌套过深，已拒绝（防绕过黑名单）"
    for text in texts:
        _interp, code = _extract_inline_code(_split_tokens(text))
        if code:
            danger = _scan_code_body(code)
            if danger:
                return danger

    # ---- 4. PowerShell -Command 正文额外扫描（增量防护）----
    # 按"任意一段的命令词"识别 PowerShell，而不是只看 heads[0]：`cmd /c powershell -c …`
    # 的外层词是 cmd，只看首词会让 PS 正文整段不扫。上面的通用扫描已经覆盖危险首词
    # 与内联解释器，这里补的是 PS 危险模式（Remove-Item -Force / iex / …）与 -Command
    # 后的正文本身。
    if any(h in ("powershell", "pwsh") for h in heads):
        # PowerShell 模式：跳过 powershell.exe [-nop] [-command] 包裹，检查 -Command 后的正文
        tokens = _split_tokens(stripped) or [stripped]
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

    return None


def _try_split_args(command: str) -> list[str] | None:
    """尝试把命令拆分成参数列表（shell=False 模式）。成功返回列表，失败返回 None。

    Windows 上有管道/重定向/拼接符时无法用参数模式执行，此时返回 None 退回 shell=True。
    """
    if _SHELL_ONLY_SIGNS.search(command):
        return None  # 有 shell 专有语法，只能用 shell=True
    args = _split_tokens(command)  # 拆词统一走 _split_tokens（ValueError 已兜成空列表）
    return [_strip_wrapping_quotes(a) for a in args] if args else None
