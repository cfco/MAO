"""覆盖写前的自动留档（data/backup/）与过期清理。

为什么独立成模块（2026-11-04，自 builtin.py 拆出守「单文件 ≤500 行」约定）：
留档机制自成一体（三道清理闸 + 留档命名），与工具定义无关。函数全部**显式收参**
（backup_dir / workspace_roots / root_resolved 由 builtin 的薄包装在调用时传入）——
测试会 monkeypatch builtin.BACKUP_DIR / ROOT_RESOLVED / _WORKSPACE_ROOTS，
薄包装每次读取模块全局再传参，patch 语义与拆分前逐字一致。
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
from pathlib import Path

BACKUP_KEEP_DAYS = 30  # 留档保留天数，超期惰性清理
MAX_BACKUP_BYTES = 1024 * 1024  # 超过该大小不做留档（避免大文件反复整份复制）
MAX_BACKUP_FILES = 300  # 留档目录文件数上限：高频覆盖场景每个 .bak 都不重名，
MAX_BACKUP_TOTAL_BYTES = 256 * 1024 * 1024  # 30 天内可无限堆积，这里按总量兜底


def purge_old_backups(
    backup_dir: Path,
    keep_days: int = BACKUP_KEEP_DAYS,
    max_files: int = MAX_BACKUP_FILES,
    max_total: int = MAX_BACKUP_TOTAL_BYTES,
) -> None:
    """清理留档目录（惰性调用，不额外起线程）。

    三道闸按序执行：① 超过保留期（30 天）的删掉；② 仍超文件数/总字节上限时，
    按 mtime 从最旧开始删到限额以内。旧实现只有第①道且**只在下一次覆盖写时触发**
    ——系统长期不覆盖文件时过期留档（含历史明文 .env 备份）会无限期躺着
    （实测 43 天未清）。除"每次覆盖写"外，build_builtin_tools 建池时也调一次，
    让清理不依赖"恰好又有覆盖写"这个低频事件。
    """
    try:
        if not backup_dir.is_dir():
            return
        cutoff = time.time() - keep_days * 86400
        kept: list[tuple[float, Path]] = []
        total = 0
        for p in backup_dir.glob("*.bak"):
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if mtime < cutoff:
                p.unlink(missing_ok=True)
                continue
            kept.append((mtime, p))
            try:
                total += p.stat().st_size
            except OSError:
                pass
        if len(kept) <= max_files and total <= max_total:
            return
        kept.sort()  # 最旧的先删
        while kept and (len(kept) > max_files or total > max_total):
            _mtime, p = kept.pop(0)
            try:
                total -= p.stat().st_size
            except OSError:
                pass
            p.unlink(missing_ok=True)
    except OSError:
        pass


def backup_before_overwrite(
    path: Path,
    backup_dir: Path,
    workspace_roots: tuple[Path, ...],
    root_resolved: Path,
    is_secret_fn,
    max_bytes: int = MAX_BACKUP_BYTES,
) -> str | None:
    """覆盖已有文件前先留档到 data/backup/，返回留档路径；无需/无法留档返回 None。

    为什么：write_file 是静默覆盖、不可逆，而项目规范要求不可逆操作可恢复
    （"删除重要文件先隔离"在覆盖场景的等价物）。工具层不引入交互式确认——
    工具调用是同步的，模型等不了人工回话；改为"自动留一份 + 如实告知位置"，
    既可回滚又不卡流程。
    """
    try:
        if not path.is_file():
            return None
        if is_secret_fn(path):
            return None  # 密钥文件不留档：否则 .env 的明文 key 会长期躺在 data/backup/ 里
        if path.stat().st_size > max_bytes:
            return None
        backup_dir.mkdir(parents=True, exist_ok=True)
        purge_old_backups(backup_dir)
        # 留档名 = 「根名前缀 + 相对路径」：分隔符换成 __ 保留层级且仍是合法文件名。
        # 外部工作区必须加根名前缀，否则不同项目里的同名文件（README.md）会在
        # data/backup/ 里互相覆盖，回滚时拿错版本。
        rel = None
        for root in workspace_roots:
            try:
                rel_path = path.relative_to(root)
            except ValueError:
                continue
            if root == root_resolved:
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
        # 路径本身再压一个短哈希：`a/b.md` 与根下真实存在的 `a__b.md` 会被上面的替换
        # 归一成同一个留档名而互相覆盖（回滚时拿错版本）；时间戳只到毫秒，MCP 宿主
        # 并发写同一文件也可能撞同一秒同名，故一并带上 pid/线程号。
        digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"{int(time.time() * 1000) % 1000:03d}"
        dest = backup_dir / f"{rel}.{digest}.{stamp}.{os.getpid()}-{threading.get_ident()}.bak"
        shutil.copy2(path, dest)
        return str(dest)
    except (OSError, ValueError):
        return None  # 留档失败不阻断写入（写入本身是可用的），但不谎报已备份
