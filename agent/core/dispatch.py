"""单工人派工：把一次请求真正打出去（含失败回退与门外重试链）。

从 orchestrator.py 拆出（2026-10-19）：orchestrator.py 单文件已达 880 行，远超仓库
「单文件 ≤500 行」约定（规则5）。单工人派工相关五个方法（ask_result / ask /
_fallback_target / _call_result / _attempt_with_retry）约 236 行，是一条聚焦「一次
派工怎么打、失败怎么回退」的完整控制流，独立成一个 Mixin。

实现方式：DispatchMixin。WorkerPool 继承它，Mixin 方法通过 self 访问 WorkerPool 的
profiles / _skip_reason / _record_result / _fallback_models / _fallback_cross_station /
_station_models，以及其它 Mixin 提供的 _chat_gated（StationGateMixin）/
_latency_sorted / _record_latency / _touch_latency（LatencyMixin）/ render_answer
（VotingMixin）。客户端缓存 `_client_for` 仍留在 orchestrator.py（池状态的一部分，
与 `_clients`/`_clients_lock` 同处），不在任何 Mixin 里。调用签名完全不变
（bridge/mcp/call 三外壳无感）。本类不单独实例化。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..config import AgentProfile
from .constants import (
    ST_COOLDOWN,
    ST_ERROR,
    ST_MISSING,
    ST_OK,
    ST_QUARANTINED,
    WORKER_SYSTEM,
)
from .llm import ERR_OTHER, LLMError, backoff_seconds
from .parallelism import StationMemo
from .redact import safe_error_text

if TYPE_CHECKING:  # 仅类型标注用，运行期不引入依赖
    from .llm import LLMClient


class DispatchMixin:
    """单工人派工能力，混入 WorkerPool 使用（本类不单独实例化）。"""

    def ask_result(
        self, worker: str, prompt: str, system: str | None = None, memo: StationMemo | None = None
    ) -> dict:
        """派工核心：返回结构化结果 {worker, ok, status, answer, error}（永不抛异常）。

        全部派工路径（ask / ask_many / collect / vote / run_review）的唯一事实源：
        成败由控制流决定（是否跳过、chat 是否抛错），而不是事后解析文本前缀（#4）。
        跳过判定（冷却 / 当日隔离）也在这里直接算出，批量路径无需再预过滤或嗅探。
        """
        p = self.profiles.get(worker)
        if not p:
            return {
                "worker": worker,
                "ok": False,
                "status": ST_MISSING,
                "answer": "",
                "error": f"子智能体 '{worker}' 不存在。可用：{', '.join(self.profiles) or '无'}",
            }
        self._touch_latency()  # 顺带检查探测是否到期（点名派工也走这里，别漏刷新延迟档案）
        reason = self._skip_reason(worker)
        if reason == "冷却中":
            return {
                "worker": worker,
                "ok": False,
                "status": ST_COOLDOWN,
                "answer": "",
                "error": f"子智能体 '{worker}' 近期连续失败，正在冷却中，建议稍后再试或换其它工人",
            }
        if reason == "当日失败隔离":
            return {
                "worker": worker,
                "ok": False,
                "status": ST_QUARANTINED,
                "answer": "",
                "error": (
                    f"子智能体 '{worker}' 今日调用已失败，当天隔离不再派工，请换其它工人或明天再试"
                ),
            }
        return self._call_result(p, system, prompt, memo=memo)

    def ask(self, worker: str, prompt: str, system: str | None = None) -> str:
        """派一个子任务给单个工人，返回其回答文本（含失败说明，不抛异常）。

        薄封装：结构化判定交给 ask_result，这里只渲染成人读文本（向后兼容）。
        免费节点友好：节点在**短时冷却**期、或当天已出现**终态不可重试错误**（当日隔离）
        时直接快速跳过并提示，不傻等；冷却到期自动回来，隔离只针对当天的坏模型。
        """
        return self.render_answer(self.ask_result(worker, prompt, system))

    def _fallback_target(
        self,
        p: AgentProfile,
        tried: set[str],
        memo: StationMemo | None = None,
    ) -> AgentProfile | None:
        """挑一个「当前可用且本轮未试过」的模型回退，供异常停止时自动恢复。

        两级候选（确定性顺序）：
        1) **同站优先**：同 (base_url, api_key) 组内、未试过、非冷却/隔离、且该站未被
           本批次探明故障（memo）的模型；
        2) **跨站兜底**（collaboration.fallback_cross_station=true 时）：同站无候选则从
           整池挑「未试过、当前可用、且其所在站未死」的模型——这就是"子智能体异常
           停止后自动换模型恢复"的落点。

        memo 为本批次共享的「死站」记忆：已探明故障的站不再重复撞（抑制额度放大）。
        候选内部再按延迟升序（问题5「优先选择延迟低的可用模型」）：同站/跨站各自排序后
        取首个；无延迟样本时排序是恒等变换，回退行为与旧版逐字一致。
        无候选返回 None（调用方不再回退）。
        """
        # 1) 同站优先
        same = [
            n
            for n in (self._station_models.get(p.station_key) or ())
            if n not in tried and n in self.profiles and self._skip_reason(n) is None
        ]
        for name in self._latency_sorted(same):
            prof = self.profiles[name]
            if memo is not None and memo.is_dead(prof.station_key):
                continue  # 该站本批次已判死，不再重复撞（抑制额度放大）
            return prof
        # 2) 跨站兜底（可关）
        if not self._fallback_cross_station:
            return None
        cross = [
            n
            for n, prof in self.profiles.items()
            if n not in tried
            and self._skip_reason(n) is None
            and not (memo is not None and memo.is_dead(prof.station_key))
        ]
        if cross:
            return self.profiles[self._latency_sorted(cross)[0]]
        return None

    def _call_result(
        self,
        p: AgentProfile,
        system: str | None,
        prompt: str,
        tried: set[str] | None = None,
        fallbacks_left: int | None = None,
        memo: StationMemo | None = None,
    ) -> dict:
        """真正打一次网络：取（或复用）LLMClient → chat → 按控制流记结构化成败。

        失败时按性质分级记账（#1/#2），并在「可重试错误」场景下自动回退换模型重试
        （见 _fallback_target）：先同站其它「当前可用」模型，同站无候选且
        collaboration.fallback_cross_station=true 时再跨站、从整池挑未试过的可用模型——
        即"子智能体异常停止就自动换模型恢复"，尽量不让一次抖动丢掉一个派工位。
        回退不重复试已试过的模型、且受 fallback_models 上限约束（抑制额度蔓延）；
        终态不可重试错误（认证等整站级问题）不回退，直接判当日隔离。

        无论成败都返回 dict、不抛异常；ask_many/collect/vote 依赖该契约。
        tried          本轮已尝试过的模型名集合（防回退环 / 重复打同一模型）
        fallbacks_left 剩余可回退次数（= collaboration.fallback_models，0=关闭回退）
        memo           本并行批次共享的「死站」记忆（见 parallelism.StationMemo）：
                       某工人的回退链因可重试失败耗尽后记下该站，同站后续工人只做那一次
                       主调用、不再各自回退撞死站，抑制 N×(1+fallback) 的跨工人额度放大。
                       顺序单发不传（None）⇒ 回退行为逐字不变。
        """
        if tried is None:
            tried = set()
        if fallbacks_left is None:
            fallbacks_left = self._fallback_models
        tried.add(p.name)
        station = p.station_key

        def _mark_station_dead() -> None:
            if memo is not None:
                memo.mark_dead(station)

        try:
            client = self._client_for(p)
            messages = [
                {"role": "system", "content": system or WORKER_SYSTEM},
                {"role": "user", "content": prompt},
            ]
            resp, err, elapsed_ms = self._attempt_with_retry(client, station, messages)
            if err is not None:
                # 分类与记账逐字沿用旧实现：LLMError 按 retryable 分级，其它意外异常按可重试
                # 处理（别全天拉黑）。回退递归发生在**站点门之外**（见 _chat_gated 文档）。
                if isinstance(err, LLMError):
                    # 可重试的瞬时错误 + 仍有回退额度 ⇒ 换模型重试（同站优先，可跨站恢复）。
                    # 死站抑制改由 _fallback_target 内的 memo 过滤承担，不再入口一刀切。
                    if err.retryable and fallbacks_left > 0:
                        fb = self._fallback_target(p, tried, memo)
                        if fb is not None:
                            # 当前模型记为瞬时失败（只冷却，当天仍可回来），换其它模型续跑
                            self._record_result(p.name, ok=False, fatal=False)
                            return self._call_result(
                                fb, system, prompt, tried, fallbacks_left - 1, memo
                            )
                    # 不再回退：按终态与否记账（可重试耗尽只冷却；终态当日隔离）。
                    # ERR_OTHER 是**兜底未知异常**（SDK 内部错、响应解析失败、环境问题等），
                    # 不代表模型本身坏了：它照旧不重试（重试无益），但**不进当日隔离**
                    # ——否则一次偶发异常就把一个健康节点封一整天。这与下面非 LLMError
                    # 分支（fatal=False，只冷却）保持同一口径。
                    if err.retryable:
                        _mark_station_dead()  # 可重试失败耗尽 ⇒ 抑制后续工人重复撞该站
                    self._record_result(
                        p.name, ok=False, fatal=not err.retryable and err.error_type != ERR_OTHER
                    )
                    return {
                        "worker": p.name,
                        "ok": False,
                        "status": ST_ERROR,
                        "answer": "",
                        "error": f"调用子智能体 '{p.name}' 失败 [{err.error_type}]："
                        f"{safe_error_text(err)}",
                    }
                if fallbacks_left > 0:
                    fb = self._fallback_target(p, tried, memo)
                    if fb is not None:
                        self._record_result(p.name, ok=False, fatal=False)
                        return self._call_result(
                            fb, system, prompt, tried, fallbacks_left - 1, memo
                        )
                _mark_station_dead()
                self._record_result(p.name, ok=False, fatal=False)
                return {
                    "worker": p.name,
                    "ok": False,
                    "status": ST_ERROR,
                    "answer": "",
                    "error": f"调用子智能体 '{p.name}' 失败："
                    f"{type(err).__name__}: {safe_error_text(err)}",
                }
            self._record_result(p.name, ok=True)
            # 真实成功派工也回填延迟档案（"一套"）：探测给小请求基线，真实调用把负载下的
            # 实际表现也计入同一份中位数——两路事实共写一个账本，选路看的是同一套数字。
            # elapsed_ms 由 _chat_gated 在**门内**计时：排队等同站前一个请求的秒数绝不能
            # 写进延迟档案，否则同站串行会被误记成"这个模型慢"，选路随之裁错。
            self._record_latency(p.name, elapsed_ms)
            return {
                "worker": p.name,
                "ok": True,
                "status": ST_OK,
                "answer": resp.get("content", "") or "(空回复)",
                "error": "",
            }
        except Exception as e:  # noqa: BLE001 - 连取 client 都失败，也不让派工崩（只冷却）
            if fallbacks_left > 0:
                fb = self._fallback_target(p, tried, memo)
                if fb is not None:
                    self._record_result(p.name, ok=False, fatal=False)
                    return self._call_result(fb, system, prompt, tried, fallbacks_left - 1, memo)
            _mark_station_dead()
            self._record_result(p.name, ok=False, fatal=False)
            return {
                "worker": p.name,
                "ok": False,
                "status": ST_ERROR,
                "answer": "",
                "error": f"调用子智能体 '{p.name}' 异常：{type(e).__name__}: {safe_error_text(e)}",
            }

    def _attempt_with_retry(
        self, client: LLMClient, station: tuple[str, str], messages: list[dict]
    ) -> tuple[dict | None, Exception | None, float]:
        """按 `llm.max_retries` 重试，但**每次尝试单独过一次站点门、退避睡眠留在门外**。

        为什么不再把 `LLMClient.chat()` 整段交给门：chat 自带退避 `sleep`，包在门里就等于
        让"等待"也占用"一站同时只跑一个模型"的配额 —— 一个坏模型会把整站锁住
        `timeout x (1+retries) + 退避` 量级，同站其它本来可用的模型全排在后面（实测第二个
        工人白等 1.55s，等的基本都是别人的 sleep）。
        重试次数与退避序列和旧实现逐字一致（`backoff_seconds` 同一个函数），只是挪到了门外。
        返回最后一次尝试的 (resp, 错误, 门内毫秒)。
        """
        attempts = 1 + self._max_retries
        resp: dict | None = None
        err: Exception | None = None
        elapsed_ms = 0.0
        for attempt in range(attempts):
            resp, err, elapsed_ms = self._chat_gated(client, station, messages)
            if err is None:
                return resp, None, elapsed_ms
            if not isinstance(err, LLMError) or not err.retryable:
                return resp, err, elapsed_ms  # 终态错误：重试也没用，直接交回上层按终态记账
            if attempt >= attempts - 1:
                return resp, err, elapsed_ms
            time.sleep(backoff_seconds(attempt, err.error_type))  # 门已释放，退避不占站配额
        return resp, err, elapsed_ms
