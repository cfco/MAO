"""运行时档案（health / latency）落盘与进程内共享的公共件。

为什么抽这一层：「锁内取快照 → 临时文件 → os.replace → 失败清 tmp」这套写法当初在
`health._write_payload`、`latency._write_payload`、`model_registry._atomic_write` 三处
逐字复制，`get_health` / `get_latency` 的进程内共享注册表又是两份。后果不只是难看：
修"旧快照覆盖新快照丢记账"这类 lost update 时必须同步改三份，漏一处就原地复发
（2026-10 之前就是这样先后打了两个同款补丁）。

口径（三条，漏一条就是事故）：
1. **取快照与写盘同在 _write_lock 之内串行**：分开取的话，两个线程"写盘顺序"可能与
   "快照顺序"相反，后写盘的拿着更早的快照把新账本整段覆盖（lost update）。
2. **落盘 IO 留在 _lock 之外**：`quarantined()` / `get_ms()` 是派工热路径，持锁写盘
   会把慢盘串进全进程派工。
3. **写失败只损失持久化**：内存账本照常工作，顺手清掉写了一半的临时文件，不留 .tmp 垃圾
   （并发 os.replace 到同一目标在 Windows 上会互相踩出残留）。
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from .redact import safe_error_text


def atomic_write_text(path: Path, text: str) -> None:
    """原子写文本：临时文件 + os.replace。任何 OSError 原样抛出，由调用方决定吞或报。

    临时名带 pid + 线程号：同进程多线程、多进程同时写同一目标时不会互踩同名 tmp。
    抛出前已命名的临时文件会被清理，不留残留。
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


class AtomicJsonLedger:
    """内存账本 + 原子落盘的公共骨架：子类只实现 `_snapshot_locked`（序列化自己的账本）。

    锁体系：`_lock` 护内存账本，`_write_lock` 串行"取快照 + 写盘"整段。
    嵌套方向恒为 write_lock → lock，且 `_lock` 临界区内绝不碰磁盘，故不会与上层
    （WorkerPool → health/latency）的取锁顺序成环。
    """

    def __init__(self, store_path: Path):
        self.store_path = Path(store_path)
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()

    def _snapshot_locked(self) -> str:  # 调用方持 _lock
        """账本 → 待写文本。子类实现，且只读内存、不碰磁盘。"""
        raise NotImplementedError

    def _disk_for_merge(self):
        """写盘前重读磁盘，返回对端进程已落盘的 (数据, 原始文本)；(None, None)=不合并。

        为什么需要：`shared_by_path` 与 `_write_lock` 的串行只在**进程内**有效。外部主常
        并发跑多个 `mao call`（每次一个新进程），各进程的内存账本互不可见，`write()` 又是
        整篇覆盖 → 后写者把先写者的记录整段抹掉（实测两个进程各记一条，6/6 轮只剩一条）。
        当日隔离记录被抹掉等于坏节点当天被反复派工，属于状态损坏而不只是"少一条样本"，
        所以需要合并；延迟账本样本会不断重测、丢一两个无害，且合并会与 `_trim` 的裁剪
        互相打架（被裁掉的样本每次又被并回来 → 档案不降反增），故保持整篇覆盖。

        第二个返回值（我们据以合并的那份原始文本）给 `write()` 做"写前重读比对"：
        读盘与落盘之间对端又写过一次的话，我们手里的快照已经过时，直接写就会把它抹掉
        （同仓 `model_registry.retire_model` 用同一招封口，这里补齐第三处）。
        """
        return None, None

    def _merge_disk_locked(self, disk) -> None:  # 调用方持 _lock
        """把 `_disk_for_merge` 读到的对端记录并进内存账本。子类实现，只读内存。"""

    def _read_disk_text(self) -> str | None:
        """只读回磁盘文本本身（给写前重读比对用）；读不到返回 None。"""
        try:
            return self.store_path.read_text(encoding="utf-8")
        except OSError:
            return None

    def write(self) -> None:
        """取快照并原子落盘（在 _lock 之外调用）。落盘失败只丢持久化，不外抛。

        顺序固定：读盘（锁外 IO）→ 合并 + 取快照（锁内纯内存）→ 落盘。
        合并过的账本在落盘前再重读一次：与合并依据不一致就整轮重来（最多 `_CAS_RETRIES`
        次），绝不拿过时快照硬覆盖对端进程刚写的记录。
        """
        with self._write_lock:
            disk, raw = self._disk_for_merge()  # IO 留在 _lock 之外
            self._ensure_loaded()  # 载入同理锁外：合并/取快照前账本必已就绪（子类实现）
            for _ in range(_CAS_RETRIES):
                with self._lock:
                    if disk is not None:
                        self._merge_disk_locked(disk)
                    text = self._snapshot_locked()
                if raw is None or self._read_disk_text() == raw:
                    break
                disk, raw = self._disk_for_merge()  # 被对端改过：拿新的一份重合并
            try:
                atomic_write_text(self.store_path, text)
            except OSError as e:
                # 写不进去只损失持久化，内存账本照常工作；但**不能一声不吭**：
                # Windows 上 os.replace 撞目标被占用是常态（杀毒/索引/另一个进程开着），
                # 静默 pass 等于档案从此不再更新而没人知道。
                _warn_write_failed(self.store_path, e)


_CAS_RETRIES = 3

# 落盘失败按路径各提示一次：长驻 bridge 里每 10 分钟一轮探测，每次都刷一行 stderr 会盖掉
# 真正的诊断（外部主读 stderr 的能力有限），但完全不报又会让"档案从此不再更新"静默发生。
_write_warned: set[str] = set()
_write_warned_lock = threading.Lock()


def _warn_write_failed(path: Path, err: OSError) -> None:
    key = str(path)
    with _write_warned_lock:
        if key in _write_warned:
            return
        _write_warned.add(key)
    print(
        f"[档案] {key} 落盘失败（{type(err).__name__}: {safe_error_text(err)}）；"
        "内存账本照常工作，但这次起的改动不会持久化（Windows 上常见原因是文件被别的进程占用）",
        file=sys.stderr,
    )


def shared_by_path(registry: dict[Path, object], lock: threading.Lock, key: Path, make) -> object:
    """按路径取进程内共享实例（同一路径全进程唯一）：避免多池各持账本互相覆写丢记账。

    `make` 只在首次创建时调用 ⇒ 其参数（registry_path / retire_days 等）先到先得，
    后续同路径调用即使参数不同也复用既有实例，保证"同一档案 = 同一个账本"的语义。
    直接构造类实例仍返回独立对象，保留"新实例=模拟新进程重读盘"的可测性。
    """
    with lock:
        inst = registry.get(key)
        if inst is None:
            inst = make()
            registry[key] = inst
        return inst
