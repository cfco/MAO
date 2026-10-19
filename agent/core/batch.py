"""批量派工：并行把同一任务打给多个工人，按入参顺序收集结果。

从 orchestrator.py 拆出（2026-10-19）：orchestrator.py 单文件已达 880 行，远超仓库
「单文件 ≤500 行」约定（规则5）。批量派工相关四个方法（collect / _gather_many /
ask_many / ask_many_structured）约 129 行，聚焦「一批怎么并发打、结果怎么按序回收并
渲染」，独立成一个 Mixin。

实现方式：BatchMixin。WorkerPool 继承它，Mixin 方法通过 self 访问 WorkerPool 的
profiles，以及其它 Mixin 提供的 _run_parallel / _effective_timeout（ParallelMixin）/
render_answer（VotingMixin）。调用签名完全不变（bridge/mcp/call 三外壳无感）。
本类不单独实例化。
"""

from __future__ import annotations

import time

from .constants import ST_COOLDOWN, ST_OK, ST_QUARANTINED, ST_TIMEOUT


class BatchMixin:
    """批量派工能力，混入 WorkerPool 使用（本类不单独实例化）。"""

    def collect(
        self,
        workers: list[str],
        prompt: str,
        system: str | None = None,
        timeout: float | None = None,
    ) -> list[tuple[str, str]]:
        """并行收集多个工人对同一任务的成功回答，按入参工人顺序返回（确定性）。

        与 ask_many 的区别：只返回成功回答的 (工人, 回答) 列表，且顺序与
        workers 入参顺序一致，不随完成先后漂移——投票编号、候选顺序
        依赖该确定性（as_completed 完成序会导致同一任务两次运行编号互换）。
        失败/冷却/当日隔离/超时/空回复的工人不进结果。

        这里**不做**冷却/隔离预过滤：判定交给 ask_result 内部，返回结构化 status，
        这里只按 status==ST_OK 过滤——成败由控制流决定，绝不解析回答文本前缀（#4）。
        """
        valid = [w for w in dict.fromkeys(workers) if w in self.profiles]
        if not valid:
            return []
        got = self._run_parallel(
            valid, prompt, system, self._effective_timeout(len(valid), timeout, valid)
        )
        return [
            (w, got[w]["answer"])
            for w in valid
            if got.get(w) is not None and got[w]["status"] == ST_OK and got[w]["answer"]
        ]

    def _gather_many(
        self,
        workers: list[str],
        prompt: str,
        system: str | None,
        timeout: float | None,
    ) -> dict:
        """并行派工的收集核心：ask_many（文本）与 ask_many_structured（结构化）共用。

        返回 {valid, got, elapsed, limit}。got[worker] 为 None 表示超时未完成；
        否则是 ask_result 的结构化 dict。elapsed[worker] 为该工人从派工到回来的毫秒数。

        不认识的工人名**不再静默丢弃**（审计 P1）：照样进入结果、由 ask_result 如实
        带回 status=missing——bridge 文档承诺的 status 词表含 missing，且外部主点名
        拼错时若只收到「全部子 AI 均未成功」，会拿着同一个错名字永远重试。
        """
        valid = list(dict.fromkeys(workers))
        if not valid:
            return {"valid": [], "got": {}, "elapsed": {}, "limit": 0.0}
        limit = self._effective_timeout(len(valid), timeout, valid)
        t0 = time.monotonic()
        elapsed: dict[str, int] = {}

        def _on_done(w: str, res: dict | None) -> None:
            # res 来自 _wait_gather 的 fut.result()：ask_result 的结构化 dict 或 None（超时）
            elapsed[w] = int((time.monotonic() - t0) * 1000)

        got = self._run_parallel(valid, prompt, system, limit, on_done=_on_done)
        return {"valid": valid, "got": got, "elapsed": elapsed, "limit": limit}

    def ask_many(
        self,
        workers: list[str],
        prompt: str,
        system: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """同一子任务并行派给多个工人，收集全部回答（失败也如实带回）。

        输出按入参工人顺序排列（确定性）；timeout 对收集阶段整体限时，缺省按
        参与人数与并发上限自适应（见 _effective_timeout）。超时工人以「未完成」带回。
        冷却/当日隔离工人不预过滤，交给 ask_result 内部判定并如实带回跳过说明。
        """
        r = self._gather_many(workers, prompt, system, timeout)
        valid = r["valid"]
        if not valid:
            return f"错误：没有可用的子智能体。可用：{', '.join(self.profiles) or '无'}"
        got, limit = r["got"], r["limit"]
        blocks: list[str] = []
        for w in valid:  # 按入参顺序输出，含跳过项
            res = got.get(w)
            if res is None:  # 整体限时到点仍未完成
                body = f"失败：超过 {limit:g}s 未完成，已放弃等待（后台线程会自行收尾）"
            elif res["status"] in (ST_COOLDOWN, ST_QUARANTINED):
                label = "冷却中" if res["status"] == ST_COOLDOWN else "当日隔离"
                body = f"错误：工人{label}，本轮跳过"
            else:  # ok 或 error：由结构化 status 渲染成人读文本，绝不回头解析文本
                body = self.render_answer(res)
            blocks.append(f"### 工人 {w} 的结果\n{body}")
        return "\n\n".join(blocks)

    def ask_many_structured(
        self,
        workers: list[str],
        prompt: str,
        system: str | None = None,
        timeout: float | None = None,
    ) -> list[dict]:
        """并行派工的结构化结果（外部主程序化消费，替代 Markdown 大块文本）。

        返回按入参顺序的 list[dict]，每项 {worker, ok(bool), status, answer, elapsed_ms}。
        status ∈ {ok, error, cooldown, quarantined, missing, timeout}（与 ask_result 同一套
        稳定码，不再翻译成中文）：直接取自结构化判定（#4，不解析文本前缀），ok 只认
        status==ok，answer 仅在成功时为正文、其余为空串。
        """
        r = self._gather_many(workers, prompt, system, timeout)
        got, elapsed = r["got"], r["elapsed"]
        out: list[dict] = []
        for w in r["valid"]:
            res = got.get(w)
            if res is None:  # 整体限时到点仍未完成
                out.append(
                    {
                        "worker": w,
                        "ok": False,
                        "status": ST_TIMEOUT,
                        "answer": "",
                        "elapsed_ms": elapsed.get(w),
                    }
                )
                continue
            out.append(
                {
                    "worker": w,
                    "ok": res["status"] == ST_OK,
                    "status": res["status"],
                    "answer": res["answer"] if res["status"] == ST_OK else "",
                    "elapsed_ms": elapsed.get(w),
                }
            )
        return out
