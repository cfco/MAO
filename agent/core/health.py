"""模型健康档案：当日失败隔离 + 连续多个"运行日"失败自动下线（在 model_registry.txt 加 #）。

规则（用户需求「失败打当天标记不再使用；连续 7 个实际工作日没运行项目的不算，都失败就标记删除」）：
- 某模型一次"终态失败"（**不可重试错误**：认证失败、非重试类 4xx 等"重试也没用"
  的调用），就往档案里该模型的 fail_dates 记一个当天日期。当天已记录 ⇒ 当天不再派工该
  模型（WorkerPool 的 ask/ask_many/collect/vote 全部跳过）。
- "运行日"（active day）= 程序实际启动并工作过的自然日，记录在档案顶层的 active_dates。
  周末没开机的日子既不算运行日、也不打断连击。
- 取最近 retire_days 个运行日（默认 7，collaboration.retire_days 可调）做窗口，
  若该模型在这些天"每天都失败过" ⇒ 自动把它在 model_registry.txt 对应 *_MODELS 行
  里的条目前加 #（下线），并在档案记 disabled 标记、stderr 提示。恢复采用人工方式：
  删掉 # 即重新上线（不自动复活，避免抖动期反复横跳）。

哪些失败**不**进这里：超时 / 限流(429) / 5xx / 连接失败等**可重试的瞬时失败**即使耗尽
重试也只在 WorkerPool 里做短时冷却（到期自动回来），不当日隔离。免费中转站天天限流抖动
是常态，一次瞬时失败就封一整天会把整池打空——故健康档案只兜"重试也没用"的硬失败。

**由此对"自动下线"的影响（务必知悉）**：既然本档案只记终态失败，那么"连续 retire_days 个
运行日都失败 → 标 # 下线"这条机制**只对重试也没用的硬失败累积生效**。一个只返回超时/429/
5xx 的模型——哪怕常年如此——不会进入本档案、也就**永远不会被自动下线**：它按设计由每次派工
的短时冷却兜底（当天被限流就退避、恢复即回来），要不要长期摘掉它交人工在 model_registry.txt
注释判断。这不是遗漏：把"临时被限流"误判成"模型已死"而自动拉黑，正是分层容错要避免的。

档案格式（collaboration.health_file，默认 data/model_health.json）：
    {"version": 2,
     "active_dates": ["2026-09-20", "2026-09-21", ...],
     "models": {"node-a:nex-n2.5-pro": {"fail_dates": ["2026-09-21", ...],
                                        "disabled": "2026-10-01" 或省略}}}
fail_dates 只保留最近 _MAX_DATES_KEPT 条、active_dates 只保留最近 _MAX_ACTIVE_KEPT
条，防止文件无界增长。旧版 version 1（无 active_dates）可读：视为运行日未知。

并发：自带独立锁（与 WorkerPool 的锁相互独立，嵌套方向恒为 pool→health，
不会反向，无死锁环）；落盘走 core/ledger.py 的原子写骨架（快照与写盘同域串行、
落盘在账本锁之外、失败只丢持久化，口径与理由见该模块文档）。写盘原子化（临时文件 + os.replace）。
离线可测：record_failure 支持显式指定日期 day，不必等真实时间流逝。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from ..model_registry import RETIRE_ALREADY, RETIRE_OFFLINED, retire_model
from .ledger import AtomicJsonLedger, shared_by_path

_MAX_DATES_KEPT = 30
_MAX_ACTIVE_KEPT = 60


def _today() -> str:
    return time.strftime("%Y-%m-%d")


class ModelHealth(AtomicJsonLedger):
    """模型健康账本：当日失败隔离查询 + 失败记账 + 连续运行日达阈值下线改写。"""

    def __init__(
        self,
        store_path: Path,
        registry_path: Path | None = None,
        retire_days: int = 7,
    ):
        # _lock / _write_lock / store_path 由基类建好（落盘口径见 ledger.py）
        super().__init__(store_path)
        self.registry_path = Path(registry_path) if registry_path else None
        self.retire_days = max(1, int(retire_days))
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

    def record_failure(
        self, name: str, model: str = "", models_env: str = "", day: str | None = None
    ) -> str | None:
        """记一次终态失败：往该模型 fail_dates 记当天日期（⇒ 当天隔离，不再派工）。

        仅由 WorkerPool 在判定为"终态不可重试失败"（auth / 非重试 4xx）时调用；可重试的
        瞬时失败由冷却承接，不会走到这里。
        name       档案键（通常是 AgentProfile.name，即池内工人的健康键）
        model      模型名（改写 model_registry.txt 时按 @ 前的名字精确匹配条目）
        models_env 该模型清单所在变量名（如 NODE_A_MODELS），定位改写行
        day        记账日期（缺省=今天；测试可注入历史日期构造连击）
        返回值：连续 retire_days 个运行日都失败并触发下线时返回提示文本（调用方打印
        stderr）；否则 None。
        """
        day = day or _today()
        notice: str | None = None
        disabled_now = False
        with self._lock:
            self._ensure_loaded()
            self._mark_active_locked(day)  # 有失败记账 ⇒ 当天必然是运行日
            entry = self._data.setdefault(name, {})
            dates = sorted(set(entry.get("fail_dates") or []) | {day})
            entry["fail_dates"] = dates[-_MAX_DATES_KEPT:]
            if not entry.get("disabled") and self._streak_full_locked(entry["fail_dates"]):
                entry["disabled"] = day
                disabled_now = True  # registry 改写是 IO，放锁外做（理由同写盘）
        # 文件 IO 放锁外：quarantined() 是派工热路径，持锁写会把 health 锁串到
        # pool 锁、卡住全进程派工。快照与落盘都在 _write_lock 内串行（见 ledger.py），
        # 这里只负责记账完就通知落盘。
        self.write()
        if disabled_now:
            notice = self._notice_and_retire(name, model, models_env)
        return notice

    def note_active(self, day: str | None = None) -> None:
        """显式登记一个运行日（测试注入用：模拟"这天开过程序、但该模型没失败"）。"""
        with self._lock:
            self._ensure_loaded()
            self._mark_active_locked(day or _today())
        self.write()  # 落盘放锁外，理由同 record_failure

    # ---------- 运行日记录 ----------

    def _mark_active_locked(self, day: str) -> None:  # 调用方持锁
        if day not in self._active:
            self._active = sorted(set(self._active) | {day})[-_MAX_ACTIVE_KEPT:]

    # ---------- 运行日连击判定 ----------

    def _streak_full_locked(self, fail_dates: list[str]) -> bool:
        """最近 retire_days 个运行日是否全部落在该模型的失败日里。

        运行日不足 retire_days 个时不判定下线（程序刚上手，样本不够）。
        周末没开机 ⇒ 那天不在 active_dates 里，既不计失败也不打断连击——
        恰好满足"7 个实际工作日、没运行项目的都不算"。
        """
        if len(self._active) < self.retire_days:
            return False
        window = self._active[-self.retire_days :]
        dset = set(fail_dates)
        return all(d in dset for d in window)

    # ---------- 自动下线改写（model_registry.txt 加 #） ----------

    def _notice_and_retire(self, name: str, model: str, models_env: str) -> str:
        """达阈值后：在 registry 给模型加 # 下线，并拼装 stderr 提示文本。"""
        hit = self._retire_in_registry(model, models_env)
        if hit:
            return (
                f"[模型健康] '{name}' 已连续 {self.retire_days} 个运行日调用失败"
                "（未启动的日子不计入、也不打断），"
                f"模型 '{model}' 已在 model_registry.txt 的 {hit} 行标 # 下线；"
                f"确认恢复免费额度/仍想使用时删掉 # 即可。"
            )
        return (
            f"[模型健康] '{name}' 已连续 {self.retire_days} 个运行日调用失败，已记入档案，"
            f"但在 model_registry.txt 的 {models_env or '对应'} 行未找到模型 '{model}' "
            "对应条目（或写入失败），请手动确认清单。"
        )

    def _retire_in_registry(self, model: str, models_env: str) -> str | None:
        """给 model_registry.txt 的 models_env 行里的 model 加 #，返回命中的变量名。

        只改写精确变量行（AgentProfile.models_env 定位，agents 池的模型才有）；
        已经带 # 的条目视为已处理（RETIRE_ALREADY，幂等：给调用方正确的"已下线"提示，
        但不重复写文件）。自动下线只作用于池模型行（models_env 非空）；无 models_env
        的键不会被改写。

        三态判定在 retire_model 内部完成——旧写法这里先自己解析一遍文件判断"是否已
        下线"、随后 retire_model 又读一遍同一文件（双份 IO + 双份解析口径，两次读之间
        文件被改还会误判），是同款逻辑复制两份的典型。
        """
        if not model or not models_env or self.registry_path is None:
            return None
        state = retire_model(self.registry_path, models_env, model)
        return models_env if state in (RETIRE_OFFLINED, RETIRE_ALREADY) else None

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
        # 注意：这里**不**把今天预记为运行日。运行日由记账决定——record_failure /
        # note_active 记哪一天，哪一天才算运行日（"没运行项目的那天不计入"）。
        # 若加载即预记今天，窗口会混入"今天还没机会失败"的日子：测试注入历史失败日
        # 会莫名多出今天、all() 判定永远失败；生产语义上第 7 个失败日到达当天也无法
        # 立刻下线，得等第 8 天。记账制下窗口紧跟真实失败节奏，更贴用户
        # 「连续 7 个实际工作日都失败 → 标 # 下线」的意图。

    def _snapshot_locked(self) -> str:  # 调用方持锁
        """把当前内存账本序列化成待写文本。锁内调用，只读内存、不碰磁盘。"""
        payload = {"version": 2, "active_dates": list(self._active), "models": self._data}
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _disk_for_merge(self):
        """写盘前重读档案，供 `_merge_disk_locked` 并上别的进程写的记录。"""
        try:
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return raw if isinstance(raw, dict) else None

    def _merge_disk_locked(self, disk: dict) -> None:  # 调用方持 _lock
        """把对端进程已落盘的记录并进内存账本（并集，不做时间仲裁）。

        为什么必须并：`get_health` 的进程内共享护不住多进程场景——外部主经常同时开几个
        `mao call`，各进程各持一份内存账本、`write()` 又整篇覆盖，实测两轮并发记账
        6/6 只剩后写那一份。被抹掉的可能是"某模型今天终态失败"的隔离记录，也可能是别人
        刚登记的运行日（→ 下线连击窗口被悄悄改写）。
        合并口径保持单调：`fail_dates`/`active_dates` 取并集后各自仍受 `_MAX_*` 裁剪，
        `disabled` 取两者中更早的那天（下线日期不该被后写者推后）。
        """
        self._ensure_loaded()
        d_models = disk.get("models")
        if isinstance(d_models, dict):
            for name, entry in d_models.items():
                if not isinstance(entry, dict):
                    continue
                mine = self._data.setdefault(name, {})
                dates = sorted(
                    set(mine.get("fail_dates") or []) | set(entry.get("fail_dates") or [])
                )
                mine["fail_dates"] = dates[-_MAX_DATES_KEPT:]
                days = [d for d in (entry.get("disabled"), mine.get("disabled")) if d]
                if days:
                    mine["disabled"] = min(days)
        d_active = [d for d in (disk.get("active_dates") or []) if isinstance(d, str)]
        if d_active:
            self._active = sorted(set(self._active) | set(d_active))[-_MAX_ACTIVE_KEPT:]

    def snapshot(self) -> dict[str, dict]:
        """当前档案的浅拷贝（诊断/测试用，不含任何密钥）。"""
        with self._lock:
            self._ensure_loaded()
            return {k: dict(v) for k, v in self._data.items()}


# ---------- 进程内共享（同一档案路径复用同一实例） ----------

_shared_lock = threading.Lock()
_shared_instances: dict[Path, ModelHealth] = {}


def get_health(
    store_path: Path, registry_path: Path | None = None, retire_days: int = 7
) -> ModelHealth:
    """按档案路径取进程内共享的 ModelHealth 实例（WorkerPool 的默认构造入口）。

    为什么必须共享：一个进程里可能有多个 WorkerPool（例如外部主在同一 bridge 进程里
    建了多套调用路径）。各持一份 ModelHealth 时，每个实例都有自己的内存账本 + 独立锁 +
    全量覆写的 write()（json.dumps(self._data)）：A 记下 node-a 今天失败并写盘后，
    B 只要持有更早的内存快照再写盘，就会把 A 的记录整段抹掉（lost update）；同时 B 的
    quarantined() 也看不到 A 的隔离结果 —— 当日隔离形同虚设，坏节点当天被反复打。共享
    同一实例后内存账本与锁唯一，写盘不再互相覆盖，隔离状态全进程一致。

    注意：直接 ModelHealth(...) 仍返回独立实例 —— 保留"新实例=模拟新进程重读盘"
    的可测性（tests/test_health.py::test_store_persists_across_instances 依赖它）。
    测试通过 MAO_HEALTH_FILE（tests/conftest.py 的 autouse 夹具）给每个用例不同
    路径，因此共享注册表不会跨用例串数据。
    registry_path / retire_days 只在**首次**创建实例时生效：同一档案路径先到先得，
    后续调用即使参数不同也复用既有实例（保证"同一档案=同一个账本"的语义不被打破）。
    """
    key = Path(store_path)
    return shared_by_path(
        _shared_instances,
        _shared_lock,
        key,
        lambda: ModelHealth(key, registry_path, retire_days),
    )
