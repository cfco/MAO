"""同站串行门（collaboration.max_per_station）：一个中转站同时最多跑 N 个在飞请求。

为什么独立成 StationGateMixin（2026-09-24）：orchestrator.py 已达 662 行、早超仓库
「单文件 ≤500 行」约定（见其文件头 TODO 口径），而"站点级并发闸门"是一条**策略**、
与"派工 + 健康 + pick 轮转"职责不同——和当初把并行段拆到 parallelism.py、投票拆到
voting.py 同理。本文件不单独实例化：方法通过 self 访问 WorkerPool 的 cfg / profiles。

策略口径（2026-09-24 定）：**一个中转站同时只跑一个模型**（默认 `max_per_station=1`）。
一站多模型只是同一份配额、同一条网关上的多个入口，并发打过去并不会更快：

1. 请求在站端排队，彼此拖慢（实测同站 3 路并发时单模型耗时数倍膨胀）；
2. 更糟的是把**延迟档案**（`_record_latency` 记的正是负载下的真实往返）污染成
   "这个模型慢"，而实际是站被自己人堵了——选路裁剪（`_latency_window`）正按这份档案
   排序，等于自己给自己制造错误的排名依据。

串行后并行度 = `min(max_workers, max_per_station × 站数)`：**跨站点名才提速**。
真实口径经 `concurrency_status()` 随 bridge 的 `health` 指令带回，主不必记忆配置。

实现要点（漏一条就会出事故）：
- 门卡在**真正发请求**的那一层（`_chat_gated` / 探测 `_probe_one`），于是批量派工、
  顺序 ask、失败回退、探测共用一个约束，不存在"批量串行但回退偷偷并发第二发"的漏口；
- 门**只包一次** chat，失败后的回退递归必须发生在门外：持着 A 站的门去等 B 站的门，
  两个工人反向回退（A→B、B→A）就会互相等成死锁；
- 排队时间不计入延迟样本（计时在门内开始），否则串行本身会伪造出"慢模型"。
"""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 仅类型标注用，运行期不引入依赖（避免 orchestrator <-> 本模块成环）
    from .llm import LLMClient


class StationGateMixin:
    """同站串行闸门能力，混入 WorkerPool 使用（本类不单独实例化）。"""

    def _init_station_gates(self, cfg) -> None:
        """建站点门表（由 WorkerPool.__init__ 调用一次）。

        门按 (base_url, api_key) 懒建，键数 = 站数（个位数量级），常驻不泄漏。
        _gates_lock 单独一把、绝不复用 self._lock：取门可能发生在任何持锁路径之外，
        而 self._lock 会被记账/pick 等逻辑持有，混用会把"排队等门"拖成"持锁等门"。
        """
        self._max_per_station = cfg.max_per_station
        self._station_gates: dict[tuple[str, str], threading.Semaphore] = {}
        self._gates_lock = threading.Lock()

    def _station_gate(self, station: tuple[str, str]) -> threading.Semaphore:
        """取（或首次建）某站的串行门。懒建 ⇒ 池里没有的站也不会凭空造门。"""
        with self._gates_lock:
            gate = self._station_gates.get(station)
            if gate is None:
                gate = threading.Semaphore(max(1, self._max_per_station))
                self._station_gates[station] = gate
            return gate

    def _chat_gated(
        self, client: LLMClient, station: tuple[str, str], messages: list[dict],
    ) -> tuple[dict | None, Exception | None, float]:
        """在「同站串行门」内发**一次**请求，返回 (resp, error, 往返毫秒)。

        - error 为 None 表示成功；异常原样带回由调用方分类（LLMError / 意外异常），
          这里不吞也不记账，保持错误分级语义在 `_call_result` 一处可见。
        - 只包一次 chat、**绝不在此递归**：见模块文档第 2 条，返回时门必定已释放。
        - 毫秒只在门内计时：等同站前一个请求的排队时间不属于这个模型的延迟。
        """
        with self._station_gate(station):
            t0 = time.monotonic()  # 只量真实请求往返（不含建连/排队），与探测同口径
            try:
                return client.chat(messages), None, (time.monotonic() - t0) * 1000.0
            except Exception as e:  # noqa: BLE001 - 交回上层按 LLMError/意外异常分类
                return None, e, (time.monotonic() - t0) * 1000.0

    def concurrency_status(self) -> dict:
        """派工前的并发口径快照：一批**真能**并行几路（同站串行下不等于 max_workers）。

        外部主靠它决定"一批点几个、要不要跨站点名、分几轮打"：只看 max_workers 会把
        3 个同站模型点成一批，实际它们是排成一路慢慢过队。
        """
        stations = len({(p.base_url, p.api_key) for p in self.profiles.values()})
        per_station = max(1, self._max_per_station)
        cap = max(1, self.cfg.max_workers)
        return {
            "stations": stations,
            "max_workers": cap,
            "max_per_station": per_station,
            "effective": max(1, min(cap, per_station * max(1, stations))),
        }
