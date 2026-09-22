"""内置工具：执行命令、文件读写、目录浏览。

安全边界：
- 文件读写/目录浏览工具的活动范围限制在「允许的根目录」内 = MAO 项目根 +
  config 的 `tools.workspace` 所列目录（默认为空 → 仅项目根，边界不变）。
  禁止通过 .. 或绝对路径穿越到这些根之外，防止会话级工具隔离被绕过。
- run_shell 有三层防护：命令黑名单拦截不可逆操作、shell 注入模式拦截、
  优先用参数模式（shell=False）避免命令拼接风险。
"""
from __future__ import annotations

import hashlib
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from ..config import ROOT, Config  # 统一从 config 拿 ROOT，避免多处重复定义漂移
from .base import FunctionTool, ToolRegistry

ROOT_RESOLVED = ROOT.resolve()  # 规范化后的根，用于越界比对

# 允许工具访问的根目录集合。默认只有 MAO 项目根；build_builtin_tools 会按
# config 的 tools.workspace 追加外部项目目录。
# 为什么用模块级而不是逐工具传参：越界判定必须在同一处生效 —— 五个工具各传一份
# 容易漏改某一个，留下越界口子。
_WORKSPACE_ROOTS: tuple[Path, ...] = (ROOT_RESOLVED,)

DEFAULT_READ_LIMIT = 2000
MAX_READ_BYTES = 64 * 1024    # read_file 单次最多读 64KB，防超长单行/超大文件
MAX_WRITE_BYTES = 1024 * 1024 # write_file / append_file 单次最多写 1MB，防误操作占满磁盘
BACKUP_DIR = ROOT / "data" / "backup"   # 覆盖写前的自动留档目录（可回滚）
BACKUP_KEEP_DAYS = 30          # 留档保留天数，超期惰性清理
MAX_BACKUP_BYTES = 1024 * 1024 # 超过该大小不做留档（避免大文件反复整份复制）

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
        for pat, desc in _PS_PATTERNS:
            if pat.search(body):
                return f"命令拦截：{desc}"
        # 兜底：正文里的 cmd 风格危险命令也拦（powershell -c "format D:"）
        for h in _command_heads(body):
            if h in _CMD_DANGEROUS_SET:
                return f"命令拦截：检测到危险命令 `{h}`"
    else:
        # cmd 或裸命令：按每段的命令首词判定（路径/包装/后缀已归一）
        for h in heads:
            if h in _CMD_DANGEROUS_SET:
                return f"命令拦截：检测到危险命令 `{h}`"

    return None


def _strip_wrapping_quotes(token: str) -> str:
    """剥掉 token 首尾成对的包裹引号（"..." 或 '...'）。

    shlex posix=False 分词会保留引号字符：`python -c "print(1)"` 会把
    带引号的表达式当参数传给 Python（静默无输出），带空格的路径同理损坏。
    引号语义已在分词时生效，这里只去掉包裹字符，恢复真实参数值。
    """
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ('"', "'"):
        return token[1:-1]
    return token


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


def _decode(b: bytes) -> str:
    """命令输出解码：优先 utf-8，失败退 gbk（Windows 中文环境），再不行丢字符。"""
    for enc in ("utf-8", "gbk"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")


def set_workspace_roots(extra: list[str] | None = None) -> tuple[Path, ...]:
    """设置允许访问的根目录 = MAO 项目根 + 配置里的额外工作区。

    为什么需要：MAO 常被用来驱动**别的项目**（在 MAO 里分析/改造另一个仓库），
    但工具层原先把路径硬绑在 MAO 自己的根上，主智能体连目标项目的文件都读不到 ——
    "驱动外部项目"就成了一句空话。

    相对路径基于 ROOT 解析；不存在或解析失败的条目直接跳过：配置写错不该让
    所有工具一起失效。
    """
    global _WORKSPACE_ROOTS
    roots: list[Path] = [ROOT_RESOLVED]
    for item in extra or []:
        text = str(item).strip()
        if not text:
            continue
        try:
            p = Path(text).expanduser()
            p = p.resolve() if p.is_absolute() else (ROOT / p).resolve()
        except (OSError, ValueError):
            continue
        if not p.is_dir():
            continue
        if p == p.parent or p == Path(p.anchor):
            # 盘符根（C:\）或文件系统根（/）：一旦成为允许的工作区，越界防护等于
            # 把整台机器交出去。anchor 判定跨平台稳妥（C:/ 的 anchor 是 C:\、
            # / 的 anchor 是 / 自身），p==p.parent 作兜底。
            print(f'[workspace] 拒绝把根目录设为工作区（会交出整机）: {p}')
            continue
        if p not in roots:
            roots.append(p)
    _WORKSPACE_ROOTS = tuple(roots)
    return _WORKSPACE_ROOTS


def _resolve(p: str) -> Path | None:
    """把用户给的路径解析为规范绝对路径；不在任何允许根目录内则返回 None。

    - 相对路径一律基于 MAO 项目根 ROOT（保持既有语义，避免含义漂移）
    - resolve() 会展开 .. 与符号链接，再逐个允许根比对父级，杜绝穿越
    - 允许的根 = ROOT + tools.workspace（默认只有 ROOT，不配即维持原安全边界）
    """
    if not isinstance(p, str) or not p.strip():
        return None
    try:
        path = Path(p).expanduser()
        abs_path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    except (ValueError, OSError):
        return None
    for root in _WORKSPACE_ROOTS:
        try:
            abs_path.relative_to(root)
            return abs_path
        except ValueError:
            continue
    return None


def _purge_old_backups() -> None:
    """清掉超过保留期的留档（惰性调用，不额外起线程）。"""
    try:
        cutoff = time.time() - BACKUP_KEEP_DAYS * 86400
        for stale in BACKUP_DIR.glob("*.bak"):
            if stale.stat().st_mtime < cutoff:
                stale.unlink(missing_ok=True)
    except OSError:
        pass


def _backup_before_overwrite(path: Path) -> str | None:
    """覆盖已有文件前先留档到 data/backup/，返回留档路径；无需/无法留档返回 None。

    为什么：write_file 是静默覆盖、不可逆，而项目规范要求不可逆操作可恢复
    （"删除重要文件先隔离"在覆盖场景的等价物）。工具层不引入交互式确认——
    工具调用是同步的，模型等不了人工回话；改为"自动留一份 + 如实告知位置"，
    既可回滚又不卡流程。
    """
    try:
        if not path.is_file():
            return None
        if path.stat().st_size > MAX_BACKUP_BYTES:
            return None
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        _purge_old_backups()
        # 留档名 = 「根名前缀 + 相对路径」：分隔符换成 __ 保留层级且仍是合法文件名。
        # 外部工作区必须加根名前缀，否则不同项目里的同名文件（README.md）会在
        # data/backup/ 里互相覆盖，回滚时拿错版本。
        rel = None
        for root in _WORKSPACE_ROOTS:
            try:
                rel_path = path.relative_to(root)
            except ValueError:
                continue
            if root == ROOT_RESOLVED:
                prefix = ""
            else:
                # 前缀 =「根名 + 根路径短哈希」：两个不同目录若恰好同名
                # （如 D:/a/project 与 C:/b/project 都叫 project），仅靠 root.name
                # 会生成同名留档互相覆盖、回滚时拿错版本；加 6 位哈希确保唯一。
                h = hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:6]
                prefix = f"{root.name}_{h}__"
            rel = prefix + rel_path.as_posix().replace("/", "__")
            break
        if rel is None:
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"{int(time.time() * 1000) % 1000:03d}"
        dest = BACKUP_DIR / f"{rel}.{stamp}.bak"
        shutil.copy2(path, dest)
        return str(dest)
    except (OSError, ValueError):
        return None  # 留档失败不阻断写入（写入本身是可用的），但不谎报已备份


def build_builtin_tools(cfg: Config) -> ToolRegistry:
    set_workspace_roots(cfg.workspace)   # 允许访问的根：MAO 项目根 + tools.workspace
    registry = ToolRegistry()
    timeout = cfg.shell_timeout

    def run_shell(args: dict) -> str:
        command = str(args.get("command", "")).strip()
        if not command:
            return "错误：command 不能为空"
        # ---- 安全检查：黑名单 + 注入模式 ----
        danger = _check_command_safety(command)
        if danger:
            return f"错误：{danger}（run_shell 已内置安全防护；如确需执行，可在项目根目录用终端手动运行）"
        # cwd 必须在项目根内且确实存在；`..`/绝对路径越界直接拒绝并给提示
        cwd_raw = str(args.get("cwd") or ".")
        cwd = _resolve(cwd_raw) if cwd_raw not in (".", ROOT.as_posix(), str(ROOT)) else ROOT_RESOLVED
        if cwd is None:
            return f"错误：cwd 超出项目根目录，被拒绝：{cwd_raw}"
        if not cwd.is_dir():
            return f"错误：cwd 不是有效目录：{cwd}"
        # ---- shell=True 加固：优先用参数模式（shell=False），不适用时退回 shell=True ----
        proc = None
        args_list = _try_split_args(command)
        try:
            if args_list is not None:
                try:
                    # 参数模式：shell=False，避免命令拼接注入
                    proc = subprocess.run(
                        args_list,
                        shell=False,
                        cwd=cwd,
                        timeout=timeout,
                        capture_output=True,
                    )
                except FileNotFoundError:
                    # 首词不是真实可执行文件 → 多半是 cmd 的**内置命令**（echo/dir/type/copy…）。
                    # 内置命令没有对应 exe，只能由 cmd 解释器执行：参数模式必然 WinError 2
                    # （实测 `echo` 在中转环境报"系统找不到指定的文件"，而工具描述明确写了支持 dir/type）。
                    # 该命令已通过黑名单与注入检查，这里退回 shell=True 不降低既有防护强度
                    # （有 `&`/`|`/`;` 的命令本来也走 shell=True）。
                    proc = subprocess.run(
                        command,
                        shell=True,
                        cwd=cwd,
                        timeout=timeout,
                        capture_output=True,
                    )
            else:
                # 有 shell 专有语法（管道/重定向等），直接用 shell=True
                proc = subprocess.run(
                    command,
                    shell=True,
                    cwd=cwd,
                    timeout=timeout,
                    capture_output=True,
                )
        except subprocess.TimeoutExpired:
            return f"错误：命令超过 {timeout} 秒未完成，已终止"
        except OSError as e:  # noqa: BLE001 - 目录等系统级错误给友好提示
            return f"错误：无法执行命令：{e}"
        parts = []
        if proc.stdout:
            parts.append(_decode(proc.stdout)[-8000:])
        if proc.stderr:
            parts.append("[stderr] " + _decode(proc.stderr)[-4000:])
        parts.append(f"[exit code] {proc.returncode}")
        return "\n".join(parts)

    def read_file(args: dict) -> str:
        p = _resolve(str(args.get("path", "")))
        if p is None:
            return f"错误：路径非法或超出项目根目录：{args.get('path', '')}"
        if not p.exists():
            return f"错误：文件不存在：{p}"
        if p.is_dir():
            return f"错误：{p} 是目录，请用 list_dir 查看目录内容"
        limit = int(args.get("limit", DEFAULT_READ_LIMIT))
        # 按字节限量读取（只读头部，不全量载入内存）：防超长单行/超大文件把内存和上下文塞爆
        with p.open("rb") as f:
            raw = f.read(MAX_READ_BYTES + 1)
        truncated_bytes = len(raw) > MAX_READ_BYTES
        raw = raw[:MAX_READ_BYTES]
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if len(lines) <= limit and not truncated_bytes:
            return text
        notes = []
        if len(lines) > limit:
            notes.append(f"共 {len(lines)} 行，展示前 {limit} 行")
        if truncated_bytes:
            notes.append("超过 64KB，已截断（仅头部）")
        return "\n".join(lines[:limit]) + f"\n...[已截断（{'；'.join(notes)}）]"

    def write_file(args: dict) -> str:
        p = _resolve(str(args.get("path", "")))
        if p is None:
            return f"错误：路径非法或超出项目根目录：{args.get('path', '')}"
        content = str(args.get("content", ""))
        if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
            return f"错误：写入内容超过 {MAX_WRITE_BYTES}B，拒绝写入（防误操作占满磁盘）"
        backup = _backup_before_overwrite(p)  # 覆盖前留档，保证可回滚
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        if backup:
            return f"已写入 {p}（原文件已留档到 {backup}，需要回滚时直接复制回来）"
        return f"已写入 {p}"

    def append_file(args: dict) -> str:
        p = _resolve(str(args.get("path", "")))
        if p is None:
            return f"错误：路径非法或超出项目根目录：{args.get('path', '')}"
        content = str(args.get("content", ""))
        if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
            return f"错误：追加内容超过 {MAX_WRITE_BYTES}B，拒绝写入（防误操作占满磁盘）"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(content)
        return f"已追加到 {p}"

    def list_dir(args: dict) -> str:
        p = _resolve(str(args.get("path", ".")))
        if p is None:
            return f"错误：路径非法或超出项目根目录：{args.get('path', '.')}"
        if not p.exists():
            return f"错误：路径不存在：{p}"
        try:
            entries = []
            for item in sorted(p.iterdir()):
                kind = "dir " if item.is_dir() else "file"
                size = "" if item.is_dir() else f"  {item.stat().st_size}B"
                entries.append(f"{kind}  {item.name}{size}")
        except PermissionError:
            return f"错误：无权限访问 {p}"
        return "\n".join(entries) if entries else "（空目录）"

    registry.register(FunctionTool(
        name="run_shell",
        description=(
            "在本机执行 shell 命令（Windows cmd 语法，如 dir、type、python、pip）。"
            "适合系统操作、批量文件处理、调用本机程序。返回 stdout/stderr 与退出码。"
            "注意：不要执行 rm/del/format 等危险或不可逆操作；"
            "工作目录限制在允许的工作区内（默认项目根，可用 config 的 tools.workspace 扩展）。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的命令"},
                "cwd": {"type": "string", "description": "工作目录，默认项目根目录"},
            },
            "required": ["command"],
        },
        func=run_shell,
    ))
    registry.register(FunctionTool(
        name="read_file",
        description="读取文本文件内容，默认最多 2000 行，超出会截断并提示。",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径（相对路径基于项目根；也可访问 tools.workspace 配置的外部目录）",
                },
                "limit": {"type": "integer", "description": "最多读取行数，默认 2000"},
            },
            "required": ["path"],
        },
        func=read_file,
    ))
    registry.register(FunctionTool(
        name="write_file",
        description=(
            "写入文本文件（覆盖），父目录不存在会自动创建。"
            "目标文件已存在时，写入前会自动留档到 data/backup/（返回文本里给出留档路径，可回滚）。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目标文件路径"},
                "content": {"type": "string", "description": "要写入的完整内容"},
            },
            "required": ["path", "content"],
        },
        func=write_file,
    ))
    registry.register(FunctionTool(
        name="append_file",
        description="向文本文件末尾追加内容。",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目标文件路径"},
                "content": {"type": "string", "description": "要追加的内容"},
            },
            "required": ["path", "content"],
        },
        func=append_file,
    ))
    registry.register(FunctionTool(
        name="list_dir",
        description="列出目录内容，标记文件/文件夹及文件大小。",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目录路径，默认项目根目录"},
            },
            "required": [],
        },
        func=list_dir,
    ))
    return registry
