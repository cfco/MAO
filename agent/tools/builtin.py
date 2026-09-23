"""内置工具：执行命令、文件读写、目录浏览。

安全边界：
- 文件读写/目录浏览工具的活动范围限制在「允许的根目录」内 = MAO 项目根 +
  config 的 `tools.workspace` 所列目录（默认为空 → 仅项目根，边界不变）。
  禁止通过 .. 或绝对路径穿越到这些根之外，防止会话级工具隔离被绕过。
- run_shell 是「命令卫生黑名单」，属**纵深防御、不是沙箱**：它有意给被沙箱限制
  的外部主一双能操作本机的"手"，真要硬隔离得靠 OS/容器层。防护三层见 shell_safety。
  启发式不可能穷尽：解释器可读写其内部路径不经文件工具的根校验，确需越界操作的
  合法场景请引导用户在终端手动执行。

文件结构：
  安全扫描（命令黑名单、注入检测、内联代码防护等）已独立到 shell_safety.py，
  本文件只保留工具定义 + 文件路径安全 + 备份机制 + run_shell 执行骨架。
"""
from __future__ import annotations

import hashlib
import subprocess  # 顶层 import：测试里会 monkeypatch bi.subprocess.run，必须让外部模块能看到
import sys
import time
from pathlib import Path

from ..config import ROOT, Config  # 统一从 config 拿 ROOT，避免多处重复定义漂移
from .base import FunctionTool, ToolRegistry, ToolResult
from .shell_safety import (  # shell 命令安全扫描，独立模块见 shell_safety.py
    _check_command_safety,
    _try_split_args,
)

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
            print(f'[workspace] 拒绝把根目录设为工作区（会交出整机）: {p}', file=sys.stderr)
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
        import shutil  # 延后 import：只有真正写文件才会用到

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
    """构建并注册全部内置工具（run_shell / read_file / write_file / append_file / list_dir）。

    所有工具共享：
    - 文件/目录路径安全（_resolve：限制在 _WORKSPACE_ROOTS 内）；
    - run_shell 前置安全检查（shell_safety._check_command_safety）。
    """
    set_workspace_roots(cfg.workspace)   # 允许访问的根：MAO 项目根 + tools.workspace
    registry = ToolRegistry()
    timeout = cfg.shell_timeout

    def run_shell(args: dict) -> str:
        """在本机执行 shell 命令（Windows cmd 语法）。

        安全加固见 shell_safety.py。参数模式（shell=False）优先，
        有 shell 专有语法（管道/重定向等）或首词是 cmd 内置命令时退回 shell=True。
        """
        command = str(args.get("command", "")).strip()
        if not command:
            return ToolResult("错误：command 不能为空", ok=False)
        # ---- 安全检查：黑名单 + 注入模式 + 重定向越界 ----
        danger = _check_command_safety(command)
        if danger:
            return ToolResult(
                f"错误：{danger}（run_shell 已内置安全防护；如确需执行，可在项目根目录用终端手动运行）",
                ok=False)
        # cwd 必须在项目根内且确实存在；`..`/绝对路径越界直接拒绝并给提示
        cwd_raw = str(args.get("cwd") or ".")
        cwd = _resolve(cwd_raw) if cwd_raw not in (".", ROOT.as_posix(), str(ROOT)) else ROOT_RESOLVED
        if cwd is None:
            return ToolResult(f"错误：cwd 超出项目根目录，被拒绝：{cwd_raw}", ok=False)
        if not cwd.is_dir():
            return ToolResult(f"错误：cwd 不是有效目录：{cwd}", ok=False)
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
            return ToolResult(f"错误：命令超过 {timeout} 秒未完成，已终止", ok=False)
        except OSError as e:  # noqa: BLE001 - 目录等系统级错误给友好提示
            return ToolResult(f"错误：无法执行命令：{e}", ok=False)
        parts = []
        if proc.stdout:
            parts.append(_decode(proc.stdout)[-8000:])
        if proc.stderr:
            parts.append("[stderr] " + _decode(proc.stderr)[-4000:])
        parts.append(f"[exit code] {proc.returncode}")
        # exit code != 0 视为命令执行失败，返回 ToolResult(ok=False) 与既有行为一致。
        # 之前这里返回裸 str——如果调用方解析 stderr 里的错误文本就能判断，但用 ok 标志更明确。
        text = "\n".join(parts)
        if proc.returncode != 0:
            return ToolResult(text, ok=False)
        return ToolResult(text, ok=True)

    def read_file(args: dict) -> str:
        p = _resolve(str(args.get("path", "")))
        if p is None:
            return ToolResult(f"错误：路径非法或超出项目根目录：{args.get('path', '')}", ok=False)
        if not p.exists():
            return ToolResult(f"错误：文件不存在：{p}", ok=False)
        if p.is_dir():
            return ToolResult(f"错误：{p} 是目录，请用 list_dir 查看目录内容", ok=False)
        # limit 是模型传参：类型错不炸工具、回退默认值（与 Config.as_int 同一态度）；
        # 负数/0 会得到"砍尾/空输出"的错误展示，钳到 >=1。
        try:
            limit = max(1, int(args.get("limit", DEFAULT_READ_LIMIT)))
        except (TypeError, ValueError):
            limit = DEFAULT_READ_LIMIT
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
            return ToolResult(f"错误：路径非法或超出项目根目录：{args.get('path', '')}", ok=False)
        content = str(args.get("content", ""))
        if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
            return ToolResult(f"错误：写入内容超过 {MAX_WRITE_BYTES}B，拒绝写入（防误操作占满磁盘）", ok=False)
        backup = _backup_before_overwrite(p)  # 覆盖前留档，保证可回滚
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        if backup:
            return f"已写入 {p}（原文件已留档到 {backup}，需要回滚时直接复制回来）"
        return f"已写入 {p}"

    def append_file(args: dict) -> str:
        p = _resolve(str(args.get("path", "")))
        if p is None:
            return ToolResult(f"错误：路径非法或超出项目根目录：{args.get('path', '')}", ok=False)
        content = str(args.get("content", ""))
        if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
            return ToolResult(f"错误：追加内容超过 {MAX_WRITE_BYTES}B，拒绝写入（防误操作占满磁盘）", ok=False)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(content)
        return f"已追加到 {p}"

    def list_dir(args: dict) -> str:
        p = _resolve(str(args.get("path", ".")))
        if p is None:
            return ToolResult(f"错误：路径非法或超出项目根目录：{args.get('path', '.')}", ok=False)
        if not p.exists():
            return ToolResult(f"错误：路径不存在：{p}", ok=False)
        try:
            entries = []
            for item in sorted(p.iterdir()):
                kind = "dir " if item.is_dir() else "file"
                size = "" if item.is_dir() else f"  {item.stat().st_size}B"
                entries.append(f"{kind}  {item.name}{size}")
        except PermissionError:
            return ToolResult(f"错误：无权限访问 {p}", ok=False)
        return "\n".join(entries) if entries else "（空目录）"

    registry.register(FunctionTool(
        name="run_shell",
        description=(
            "在本机执行 shell 命令（Windows cmd 语法，如 dir、type、python、pip）。"
            "适合系统操作、批量文件处理、调用本机程序。返回 stdout/stderr 与退出码。"
            "注意：内置危险命令拦截（rm/del/format 等，且 python -c / node -e 这类内联代码"
            "正文也会扫描破坏性调用），但这是命令卫生黑名单、**不是沙箱**；确需的不可逆操作"
            "请引导用户用终端手动执行。工作目录限制在允许的工作区内（默认项目根，"
            "可用 config 的 tools.workspace 扩展）。"
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
