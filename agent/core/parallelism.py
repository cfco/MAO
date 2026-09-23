"""并行派工：线程 + 整体限时收集（从 orchestrator.py 二次拆分，2026-09-23）。

为什么独立成 ParallelMixin：orchestrator.py 在 2026-09-22 把投票拆到 voting.py 后
仍有 543 行，超「单文件 ≤500 行」约定。并行派工相关四个方法（_wait_gather /
_effective_timeout / _spawn / _run_parallel）约 100 行，与「派工 + 健康 + pick」
职责不同，独立成 Mixin 后——
  - orchestrator.py 回到 ~443 行，专注池 / 派工 / 健康 / pick 轮转；
  - parallelism.py 专注「怎么把一批 ask_result 并发打出去并限时收回」。

方法通过 self 访问 WorkerPool 的 cfg / ask_result / max_workers 等，与 VotingMixin
同模式：本类不单独实例化，由 WorkerPool(ParallelMixin, VotingMixin) 提供 self 上下文。
"""
from __future__ import annotations

import concurrent.futures
import threading
import time
from concurrent.futures import Future, wait

from .constants import ST_ERROR


class ParallelMixin:
    """并行派工能力，混入 WorkerPool 使用。

    本类不单独实例化：所有方法都依赖 self 上存在 cfg / ask_result / max_workers 等
    属性与方法，这些由 WorkerPool 提供。定义为 Mixin 只是为了让 orchestrator.py 更短。
    """

    # ---------- 并行派工 ----------

    def _wait_gather(
        self, futs: dict[Future, str], timeout: float, on_done=None,
    ) -> dict[str, dict | None]:
        """等待一批 {Future: 工人名}，整体限时，返回 {工人: 结果 dict 或 None}。

        每个值是 ask_result 的结构化 dict；None 表示整体限时到点仍未完成（调用方据此
        标 timeout）。用 `wait(..., FIRST_COMPLETED)` 边完成边收集；`on_done(worker, res)`
        每有一个工人回来就回调一次（ask_many 用它记耗时）。回调异常被吞。
        ask_result 本身不抛，这里的 except 只是兜底：万一抛了也归一成 error dict。
        """
        fut_to_worker = futs
        deadline = time.time() + timeout
        out: dict[str, dict | None] = {}
        pending = set(fut_to_worker)
        while pending:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            done, pending = wait(
                pending, timeout=min(remaining, 0.5),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                w = fut_to_worker[fut]
                try:
                    res = fut.result()
                except Exception as e:  # noqa: BLE001 - 兜底归一成结构化 error，绝不下游解析文本
                    res = {"worker": w, "ok": False, "status": ST_ERROR, "answer": "",
                           "error": f"派工异常：{type(e).__name__}: {e}"}
                out[w] = res
                if on_done is not None:
                    try:
                        on_done(w, res)
                    except Exception:  # noqa: BLE001 - 回调异常不影响派工
                        pass
        for fut in pending:  # 超时：放弃等待（已启动的后台自行收尾）
            fut.cancel()
            out[fut_to_worker[fut]] = None
        return out

    def _effective_timeout(self, n: int, timeout: float | None) -> float:
        """批量派工的整体限时：显式传入优先，缺省按「批次 × 单请求超时」推导。

        为什么不用固定 300s：批次 = ceil(n / max_workers)，每批最长 llm.timeout 秒。
        池大 + 并发低时（实测 25 个工人 / 3 并发 = 9 批），固定限时在单工人耗时超过
        「限时 ÷ 批次」秒时（>33s 就触发）会把后面所有批次判为"未完成"丢弃 ——
        票和草稿都拿不全，额度却已经花掉。自适应后限时随规模伸缩，并加 30s 余量
        给排队与序列化开销；配合 pick() 的参与人数上限，规模本身也回到可控区间。
        """
        if timeout is not None:
            return float(timeout)
        per_call = self.cfg.as_float(
            self.cfg.llm_cfg.get("timeout", 120) or 120, "llm.timeout", 120.0, minimum=1.0
        )
        batches = max(1, -(-max(1, n) // self.cfg.max_workers))
        return batches * per_call + 30.0

    def _spawn(self, fn, sem: threading.Semaphore) -> Future:
        """起一个 daemon 工作线程执行 fn，返回其 Future（取代 ThreadPoolExecutor）。

        为什么不用 ThreadPoolExecutor（审计 P2）：它的 worker 是非 daemon 线程，
        解释器退出时 atexit 会把它们全部 join——批量派工里的「放弃等待」只是不再
        等结果，bridge/mcp 进程收尾仍会被在飞请求（最长 LLM timeout×retries）拖住。
        daemon 线程让退出干脆。取消语义保留：整体限时到点后尚未开跑的任务被 cancel，
        set_running_or_notify_cancel 返回 False ⇒ 不再打网络，不白烧免费额度。
        """
        fut: Future = Future()

        def _run() -> None:
            with sem:
                if not fut.set_running_or_notify_cancel():
                    return  # 已被放弃（cancel），不派工
                try:
                    fut.set_result(fn())
                except BaseException as e:  # noqa: BLE001 - 异常必须进 future，不能杀线程
                    fut.set_exception(e)

        threading.Thread(target=_run, daemon=True).start()
        return fut

    def _run_parallel(
        self, workers: list[str], prompt: str, system: str | None, timeout: float,
        on_done=None,
    ) -> dict[str, dict | None]:
        """并行派工给多个工人并限时收集；未完成的放弃等待（daemon 线程，不拖退出）。

        并发单元固定为 ask_result（结构化、不抛异常），也是各类 spy 测试的 patch 点。
        并发上限由信号量把守（与原线程池容量等价），批次节奏与 _effective_timeout
        的推导保持一致。on_done 透传给 _wait_gather（ask_many 用它记每个工人耗时；
        collect/vote 不传）。
        """
        if not workers:
            return {}
        sem = threading.Semaphore(min(len(workers), self.cfg.max_workers))
        futs = {
            self._spawn(lambda w=w: self.ask_result(w, prompt, system), sem): w
            for w in workers
        }
        return self._wait_gather(futs, timeout, on_done=on_done)
