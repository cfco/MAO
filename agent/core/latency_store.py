"""延迟账本（LatencyStore）：每个工人的最近往返延迟样本 → 中位数，落盘与进程内共享。

为什么独立成模块（2026-11-04，自 latency.py 拆出守「单文件 ≤500 行」约定）：账本
（存取/落盘/共享）与探测调度（LatencyProber）、选路裁剪（LatencyMixin）是三种职责，
拆开后 latency.py 专注"怎么测、怎么用"，本模块专注"怎么存"。

与 health.py 同一套纪律：写盘前重读合并、锁内不碰磁盘、按路径进程内共享
（get_latency）——多池各持账本会互相覆写丢更新。落盘只记延迟数值，不含任何密钥。
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from pathlib import Path

from .ledger import AtomicJsonLedger, shared_by_path

_MAX_SAMPLES = 5  # 每工人保留最近 N 次延迟样本（中位数抗单次抖动）
_VIEW_KEEP = 500  # 档案最多保留的工人条数（防异常膨胀）


def _median(samples: list) -> float | None:
    """样本中位数；空 / 全非法返回 None（不抛异常）。"""
    nums: list[float] = []
    for s in samples:
        try:
            nums.append(float(s))
        except (TypeError, ValueError):
            continue
    if not nums:
        return None
    return round(statistics.median(nums), 1)


def _trim(data: dict[str, dict]) -> dict[str, dict]:
    """只保留最近更新的 _VIEW_KEEP 个工人条目（按 updated_at），防档案无限膨胀。"""
    if len(data) <= _VIEW_KEEP:
        return data
    items = sorted(data.items(), key=lambda kv: kv[1].get("updated_at", 0.0))
    return dict(items[-_VIEW_KEEP:])


class LatencyStore(AtomicJsonLedger):
    """模型延迟账本：最近 N 次往返延迟样本 → 中位数。热路径只读内存。"""

    def __init__(self, store_path: Path):
        # _lock / _write_lock / store_path 与落盘口径由基类提供（见 ledger.py）
        super().__init__(store_path)
        self._data: dict[str, dict] | None = None  # 懒加载：首次使用时才读盘
        self._load_lock = threading.Lock()  # 只护"首次读盘"，与 _lock 分开（见 _ensure_loaded）
        self._snapshot_cache: dict[str, float] | None = None  # 中位数结果缓存（见 snapshot）

    # ---------- 查询 ----------

    def get_ms(self, name: str) -> float | None:
        """该工人当前延迟（最近样本中位数，毫秒）；无样本返回 None。只查内存。"""
        self._ensure_loaded()  # 读盘放在 _lock 之外：载入不在记账锁内碰磁盘
        with self._lock:
            entry = self._data.get(name) or {}
            return _median(entry.get("samples") or [])

    def snapshot(self) -> dict[str, float]:
        """{worker: 延迟中位数ms}（只含有样本的工人），供 pick 排序与诊断。

        中位数结果带缓存：pick 每次派工都会调它，而样本只有在 record/record_many 时才变
        （那时缓存随之作废），没必要每次把全池样本重算一遍中位数。返回一份拷贝，
        调用方可随意读。
        """
        if self._snapshot_cache is not None:
            return dict(self._snapshot_cache)
        self._ensure_loaded()  # 读盘放在 _lock 之外
        with self._lock:
            out: dict[str, float] = {}
            for k, v in self._data.items():
                ms = _median(v.get("samples") or [])
                if ms is not None:
                    out[k] = ms
            self._snapshot_cache = out
        return dict(out)

    # ---------- 记录 ----------

    def record(self, name: str, ms: float) -> None:
        """记一次往返延迟（毫秒）：追加样本、保留最近 _MAX_SAMPLES 条，再落盘。

        落盘放在锁外（自带 _write_lock）：get_ms/snapshot 是 pick 热路径，绝不能被磁盘
        写拖住。写入失败只损失持久化，内存账本照常工作。
        """
        self.record_many([(name, ms)])

    def record_many(self, samples: list[tuple[str, float]]) -> None:
        """批量记一批样本，**只落盘一次**（探测轮用：N 个工人各写一遍整份档案就是 N 次 IO）。

        与 record 同一口径：单条非法值（非数字 / 负数）跳过该条、其余照记；一条都没记成
        就不落盘。每工人保留最近 _MAX_SAMPLES 条；落盘在锁外（见 ledger.write）。
        """
        self._ensure_loaded()  # 读盘放在 _lock 之外
        added = False
        with self._lock:
            for name, ms in samples:
                try:
                    value = float(ms)
                except (TypeError, ValueError):
                    continue
                if value < 0:
                    continue
                entry = self._data.setdefault(name, {})
                arr = list(entry.get("samples") or [])
                arr.append(round(value, 1))
                entry["samples"] = arr[-_MAX_SAMPLES:]
                entry["updated_at"] = time.time()
                added = True
            if added:
                self._data = _trim(self._data)
                self._snapshot_cache = None  # 样本变了：作废中位数缓存
        if added:
            self.write()  # 快照+落盘都在 _write_lock 内串行，见 ledger.AtomicJsonLedger.write

    # ---------- 落盘 ----------

    def _ensure_loaded(self) -> None:
        """保证内存账本已载入盘上数据。**必须在 _lock 之外调用**（载入要读盘）。

        持 _lock 读盘会把慢盘串进 pick 热路径（ledger.AtomicJsonLedger 的"锁内绝不碰磁盘"
        口径）。这里不碰 _lock，只用 _load_lock 防并发首查各读一遍盘；已载入的快速路径
        连 _load_lock 都不取。写盘路径由基类 write() 保证先载入再进 _lock。
        """
        if self._data is not None:
            return
        with self._load_lock:
            if self._data is not None:
                return
            data: dict[str, dict] = {}
            try:
                raw = json.loads(self.store_path.read_text(encoding="utf-8"))
                models = raw.get("models") if isinstance(raw, dict) else None
                if isinstance(models, dict):
                    data = {k: v for k, v in models.items() if isinstance(v, dict)}
            except (OSError, ValueError):
                data = {}  # 缺失/损坏都从空白账本重新开始，不拖垮派工
            self._data = data

    def _snapshot_locked(self) -> str:  # 调用方持锁
        """内存账本 → 待写文本。锁内只读内存、不碰磁盘。"""
        payload = {"version": 1, "models": self._data}
        return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------- 进程内共享（同一档案路径复用同一实例） ----------

_shared_lock = threading.Lock()
_shared_instances: dict[Path, LatencyStore] = {}


def get_latency(store_path: Path) -> LatencyStore:
    """按档案路径取进程内共享的 LatencyStore（WorkerPool 默认构造入口）。

    与 get_health 同理：同进程多 WorkerPool 时共享同一账本，避免各自覆写丢更新。
    直接 LatencyStore(...) 仍返回独立实例，保留"新实例=模拟新进程重读盘"的可测性。
    """
    key = Path(store_path)
    return shared_by_path(_shared_instances, _shared_lock, key, lambda: LatencyStore(key))
