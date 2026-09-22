"""模型健康档案：当日失败隔离（一次终态不可重试失败 → 当天不再派工该模型）。

规则：某模型出现"终态失败"（**不可重试错误**：认证失败、非重试类 4xx 等"重试也没用"
的调用），就往档案里该模型的 fail_dates 记一个当天日期。当天已记录 ⇒ 当天不再派工该
模型（WorkerPool 的 ask/ask_many/collect/vote 全部跳过）。

哪些失败**不**进这里：超时 / 限流(429) / 5xx / 连接失败等**可重试的瞬时失败**即使耗尽
重试也只在 WorkerPool 里做短时冷却（到期自动回来），不当日隔离。免费中转站天天限流抖动
是常态，一次瞬时失败就封一整天会把整池打空——故健康档案只兜"重试也没用"的硬失败。

（历史上的"连续 N 个运行日失败就自动在 .env.example 给模型加 # 下线"机制已移除——
按需驱动下路由取舍交给外部主，静默改配置文件反而添乱；只保留当日失败隔离。）

档案格式（collaboration.health_file，默认 data/model_health.json）：
    {"version": 2,
     "active_dates": ["2026-09-20", "2026-09-21", ...],
     "models": {"node-a:nex-n2.5-pro": {"fail_dates": ["2026-09-21", ...]}}}
fail_dates 只保留最近 _MAX_DATES_KEPT 条、active_dates 只保留最近 _MAX_ACTIVE_KEPT
条，防止文件无界增长。active_dates 目前仅作运行痕迹记录（自动下线移除后不再参与判定）。

并发：自带独立锁（与 WorkerPool 的锁相互独立，嵌套方向恒为 pool→health，
不会反向，无死锁环）；写盘原子化（临时文件 + os.replace）。
离线可测：record_failure 支持显式指定日期 day，不必等真实时间流逝。
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

_MAX_DATES_KEPT = 30
_MAX_ACTIVE_KEPT = 60


def _today() -> str:
    return time.strftime("%Y-%m-%d")


class ModelHealth:
    """模型健康账本：当日失败隔离查询 + 失败记账。"""

    def __init__(
        self,
        store_path: Path,
    ):
        self.store_path = Path(store_path)
        self._lock = threading.Lock()
        # 落盘专用锁：写盘已移到 _lock 之外（见 _write_payload），多个线程可能同时
        # 落盘 —— 需要串行化，否则并发 os.replace 到同一目标在 Windows 上会互相踩。
        self._write_lock = threading.Lock()
        self._data: dict[str, dict] | None = None  # 懒加载：首次使用时才读盘
        self._active: list[str] = []  # 运行日清单（加载后随记账增长）

    # ---------- 查询 ----------

    def quarantined(self, name: str) -> bool:
        """当天记过失败 ⇒ 隔离（当天不再派工）。只查内存，热路径友好。"""
        with self._lock:
            self._ensure_loaded()
            entry = self._data.get(name) or {}
            return _today() in (entry.get("fail_dates") or [])

    # ---------- 记账 ----------

    def record_failure(self, name: str, model: str = "", models_env: str = "",
                       day: str | None = None) -> str | None:
        """记一次终态失败：往该模型 fail_dates 记当天日期（⇒ 当天隔离，不再派工）。

        仅由 WorkerPool 在判定为"终态不可重试失败"（auth / 非重试 4xx）时调用；可重试的
        瞬时失败由冷却承接，不会走到这里。
        name 档案键（通常是 AgentProfile.name，主智能体兜底模型用 solo:<model>）
        model / models_env 曾用于自动改写 .env.example 下线，该机制已移除，参数保留
        仅为兼容既有调用方。day 供测试注入历史日期。恒返回 None（不再产生下线提示）。
        """
        day = day or _today()
        with self._lock:
            self._ensure_loaded()
            self._mark_active_locked(day)  # 有失败记账 ⇒ 当天必然是运行日
            entry = self._data.setdefault(name, {})
            dates = sorted(set(entry.get("fail_dates") or []) | {day})
            entry["fail_dates"] = dates[-_MAX_DATES_KEPT:]
            payload = self._snapshot_locked()
        # 文件 IO 放锁外：quarantined() 是派工热路径，持锁写会把 health 锁串到
        # pool 锁、卡住全进程派工。锁内只更新内存并取快照，写盘用快照文本。
        self._write_payload(payload)
        return None

    def note_active(self, day: str | None = None) -> None:
        """显式登记一个运行日（测试注入用：模拟"这天开过程序、但该模型没失败"）。"""
        with self._lock:
            self._ensure_loaded()
            self._mark_active_locked(day or _today())
            payload = self._snapshot_locked()
        self._write_payload(payload)  # 文件 IO 放锁外，理由同 record_failure

    # ---------- 运行日记录 ----------

    def _mark_active_locked(self, day: str) -> None:  # 调用方持锁
        if day not in self._active:
            self._active = sorted(set(self._active) | {day})[-_MAX_ACTIVE_KEPT:]

    # ---------- 落盘 ----------

    def _ensure_loaded(self) -> None:  # 调用方持锁
        if self._data is not None:
            return
        data: dict[str, dict] = {}
        active: list[str] = []
        try:
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
            models = raw.get("models") if isinstance(raw, dict) else None
            if isinstance(models, dict):
                data = {k: v for k, v in models.items() if isinstance(v, dict)}
            if isinstance(raw, dict):
                active = [d for d in (raw.get("active_dates") or []) if isinstance(d, str)]
        except (OSError, ValueError):
            data = {}  # 缺失/损坏都从空白账本重新开始，不拖垮派工
        self._data = data
        self._active = sorted(set(active))[-_MAX_ACTIVE_KEPT:]
        # 能加载到这一步说明程序今天在工作：当天先记为运行日。
        # 不立刻写盘（查询路径也走这里，避免只读操作产生副作用），
        # 由后续记账/登记的 _write_payload 一并落盘。
        self._mark_active_locked(_today())

    def _snapshot_locked(self) -> str:  # 调用方持锁
        """把当前内存账本序列化成待写文本。锁内调用，只读内存、不碰磁盘。"""
        payload = {"version": 2, "active_dates": list(self._active), "models": self._data}
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _write_payload(self, text: str) -> None:
        """把快照原子写入档案。**锁外调用**，不阻塞 quarantined() 等热路径。

        自己带一把写盘锁（_write_lock）串行化落盘：写盘移出 _lock 后多个线程可能
        同时写，并发 os.replace 到同一目标在 Windows 上会互相踩（实测残留 .tmp
        垃圾文件）。临时文件名仍带 pid+线程号，失败时清理，不留残留。
        """
        tmp: Path | None = None
        with self._write_lock:
            try:
                self.store_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.store_path.with_name(
                    f"{self.store_path.name}.{os.getpid()}-{threading.get_ident()}.tmp"
                )
                tmp.write_text(text, encoding="utf-8")
                os.replace(tmp, self.store_path)
            except OSError:
                # 档案写不进去只损失持久化，内存账本照常工作；顺手清掉写了一半的临时文件
                if tmp is not None:
                    try:
                        tmp.unlink(missing_ok=True)
                    except OSError:
                        pass

    def snapshot(self) -> dict[str, dict]:
        """当前档案的浅拷贝（诊断/测试用，不含任何密钥）。"""
        with self._lock:
            self._ensure_loaded()
            return {k: dict(v) for k, v in self._data.items()}


# ---------- 进程内共享（同一档案路径复用同一实例） ----------

_shared_lock = threading.Lock()
_shared_instances: dict[Path, ModelHealth] = {}


def get_health(store_path: Path) -> ModelHealth:
    """按档案路径取进程内共享的 ModelHealth 实例（WorkerPool 的默认构造入口）。

    为什么必须共享：一个进程里可能有多个 WorkerPool（例如外部主在同一 bridge 进程里
    建了多套调用路径）。各持一份 ModelHealth 时，每个实例都有自己的内存账本 + 独立锁 +
    全量覆写的 _write_payload（json.dumps(self._data)）：A 记下 node-a 今天失败并写盘后，
    B 只要持有更早的内存快照再写盘，就会把 A 的记录整段抹掉（lost update）；同时 B 的
    quarantined() 也看不到 A 的隔离结果 —— 当日隔离形同虚设，坏节点当天被反复打。共享
    同一实例后内存账本与锁唯一，写盘不再互相覆盖，隔离状态全进程一致。

    注意：直接 ModelHealth(...) 仍返回独立实例 —— 保留"新实例=模拟新进程重读盘"
    的可测性（tests/test_health.py::test_store_persists_across_instances 依赖它）。
    测试通过 MAO_HEALTH_FILE（tests/conftest.py 的 autouse 夹具）给每个用例不同
    路径，因此共享注册表不会跨用例串数据。
    """
    key = Path(store_path)
    with _shared_lock:
        inst = _shared_instances.get(key)
        if inst is None:
            inst = ModelHealth(key)
            _shared_instances[key] = inst
        return inst
