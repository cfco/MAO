"""投票与共识：从多个工人候选方案中挑出最优解。

本模块从 orchestrator.py 拆出（2026-09-22），原因：
- 当时 WorkerPool 单文件 543 行，超 500 行约定；投票相关方法（vote / select_best /
  _run_ballot / render_answer）约 130 行，独立后 orchestrator.py 专注派工与并行，
  voting.py 专注"从候选中挑出最好"的决策逻辑，职责更清晰。
- 后续（2026-09-23）并行派工段（_wait_gather / _effective_timeout / _spawn /
  _run_parallel）进一步拆到 parallelism.py，orchestrator.py 已回到 ≤500 行约束。
- 投票逻辑的复用面（将来外部主 AI 也可能直接调用 select_best 做评审候选）
  比派工更广，独立模块后 import 路径更直观。

实现方式：VotingMixin。WorkerPool 继承它，Mixin 方法通过 self 访问 WorkerPool
的 cfg / profiles / pick / collect / _run_parallel / _effective_timeout 等属性与方法，
调用签名完全不变（外部 bridge/mcp/call 三外壳无感）。
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .constants import ST_OK, VOTER_SYSTEM

if TYPE_CHECKING:  # 仅类型标注用，避免循环 import
    from .orchestrator import WorkerPool


class VotingMixin:
    """投票与择优能力，混入 WorkerPool 使用。

    本类**不**单独实例化——所有方法都依赖 self 上存在 cfg / profiles / health /
    pick / collect / _run_parallel / _effective_timeout 等属性与方法，这些由
    WorkerPool 提供。定义为 Mixin 只是为了让 orchestrator.py 更短。
    """

    # ---------- render ----------

    @staticmethod
    def render_answer(r: dict) -> str:
        """把结构化派工结果渲染成对外文本（单 ask / ask_many 的人读输出）。

        成败只认 status，绝不回头解析回答文本里的「错误：」字样——单一事实源：
        ask()/ask_many 的人读文本都由这里生成，内部判定一律走 ask_result 的结构。
        """
        if r["status"] == ST_OK:
            return r["answer"]
        return f"错误：{r['error']}"

    # ---------- 投票核心 ----------

    def _run_ballot(
        self: WorkerPool,
        candidates: list[tuple[str, str]],
        voters: list[str],
        timeout: float | None = None,
    ) -> tuple[int, dict[int, int], int, int, int] | None:
        """对编号候选方案发起投票（两步投票的第二步）。

        返回 (best_idx, tally, total, best_n, invalid)：total 是**有效票数**，
        invalid 是投了不存在编号的废票数（调用方据此如实提示）。
        候选编号与 candidates 下标一一对应（candidates 顺序由调用方保证确定性）。
        无任何可解析选票（超时/工人都没输出编号）返回 None。
        阈值不在这里判（共识与否由调用方按得票比例决定），故本函数不收阈值参数。
        """
        numbered = [
            f"[{i}] {w}\n预览：{ans[:80]}".replace("\n", " ")
            for i, (w, ans) in enumerate(candidates, 1)
        ]
        ballot = "候选方案：\n" + "\n".join(numbered) + "\n\n请只输出你认可方案的编号数字。"
        got = self._run_parallel(
            voters, ballot, VOTER_SYSTEM, self._effective_timeout(len(voters), timeout)
        )
        votes: list[int] = []
        for res in got.values():
            # 只有成功派工的 ST_OK 回答才算"投票意见"：错误/跳过/超时按结构化 status
            # 排除，绝不解析文本前缀（#4）。ask_result 已保证失败时 answer="" —— 双重保险，
            # 冷却提示、HTTP 状态码、"超过 Ns 未完成"里带的数字都不会被误当成选票。
            if res is None or res["status"] != ST_OK:
                continue
            m = re.search(r"(?<!\d)\d+(?!\d)", res["answer"])
            if m:
                votes.append(int(m.group()))
        if not votes:
            return None
        # 分母只算**有效票**：模型偶尔会随口给个不存在的编号（如候选只有 3 份却投 9），
        # 这种废票不该出现在共识比例的分母里。原实现 total = len(votes) 把废票也算进
        # 分母，会出现"有效票 1/1 全投方案一、却被另 2 张废票稀释成 1/3 判为未达共识"
        # 的错误结论 —— 投票是"取共识"的关键路径，分母错就等于结论错。
        valid = [v for v in votes if 1 <= v <= len(candidates)]
        if not valid:
            return None
        total = len(valid)
        invalid = len(votes) - total
        tally: dict[int, int] = {}
        for v in valid:
            tally[v] = tally.get(v, 0) + 1
        best_idx, best_n = max(tally.items(), key=lambda kv: kv[1])
        return best_idx, tally, total, best_n, invalid

    def select_best(
        self: WorkerPool,
        candidates: list[tuple[str, str]],
        voters: list[str] | None = None,
        threshold: float | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str, str] | None:
        """对已有候选方案投票择优（复用两步投票的第二步，不重复收集阶段）。

        返回 (胜出工人名, 胜出全文, 票况摘要)；无解析选票返回 None。
        供调用方「候选已就绪、只想复用投票择一」的场景（程序化公共 API，见 WorkerPool 类文档）。
        voters 缺省取 pick() 的可用工人子集（受 collaboration.max_participants 约束）。
        """
        # 审计 P1：显式传入的 threshold 也过 as_float——外部主经 bridge 传 JSON 时
        # "threshold":"0.5"（字符串）会原样流到 `ratio >= th` 抛 TypeError。
        # 与数值配置同一态度：类型错警告一次、回退默认值，不阻投票。
        th = self.cfg.as_float(
            self.cfg.collab_cfg.get("vote_threshold", 0.5) if threshold is None else threshold,
            "collaboration.vote_threshold", 0.5, minimum=0.0,
        )
        pool = [w for w in dict.fromkeys(voters or self.pick()) if w in self.profiles]
        if not pool:
            return None
        result = self._run_ballot(candidates, pool, timeout)
        if result is None:
            return None
        best_idx, tally, total, best_n, _invalid = result
        winner = candidates[best_idx - 1]
        summary = (
            f"投票 {best_n}/{total} 票（阈值 {th:.0%}）；票况："
            + ", ".join(f"[{i}]={tally.get(i, 0)}" for i in range(1, len(candidates) + 1))
        )
        return winner[0], winner[1], summary

    def vote(
        self: WorkerPool,
        prompt: str,
        workers: list[str] | None = None,
        threshold: float | None = None,
        timeout: float | None = None,
    ) -> dict:
        """两步投票：先让每个工人给出方案，再让全体工人对方案编号投票。

        投票规则：
        1) 并行收集各工人对同一任务的目标作答（成功结果才进入候选池），
           候选顺序按入参工人顺序排列（编号确定性，重跑不互换）。
        2) 把候选方案编号（附 80 字符预览）重新发给全体工人，每人选一个编号。
        3) 统计得票：最高票数 / 总票数 >= 阈值（默认 collaboration.vote_threshold，
           通常取 0.5，即过半共识）则宣布达成共识，否则列出票况请主智能体裁决。

        workers 缺省取 pick()（按 collaboration.max_participants 限制参与人数、
        跳过冷却与当日隔离的工人）—— 全池投票在「一站多模型」的池子里意味着
        收集 N 次 + 投票 N 次共 2N 个请求，免费额度扛不住且尾部批次必然超时。

        timeout 语义（审计遗留项收口）：这里把显式传入的 timeout 当作**整次投票
        的总预算**，收集与投票两阶段各分一半——调用方说 120s 就是整次投票最迟
        120s 收工。旧实现把同一个 timeout 连用两遍（每阶段都限时 timeout），
        整体最长 ≈ 2×timeout，与 REQUIREMENTS §4 对 timeout 的描述（批量派工
        限时）口径不一致。缺省（不传）时每阶段仍各自按 _effective_timeout 自适应，
        不受此影响。

        返回结构化 dict（#2：让外层 ok 反映真实成败，不再只回一段文本）：
          {ok: bool, consensus: bool, report: str}
          · ok=False：无法投票（没有可投票工人 / 全员无可用方案 / 结果无法解析），
            report 为原因文本；
          · ok=True & consensus=True：达成过半共识，report 含胜出方案全文；
          · ok=True & consensus=False：投票成功但未过半，report 列票况请主裁决。
        """
        # 审计 P1：显式传入的 threshold 也过 as_float——外部主经 bridge 传 JSON 时
        # "threshold":"0.5"（字符串）会原样流到 `ratio >= th` 抛 TypeError。
        # 与数值配置同一态度：类型错警告一次、回退默认值，不阻投票。
        th = self.cfg.as_float(
            self.cfg.collab_cfg.get("vote_threshold", 0.5) if threshold is None else threshold,
            "collaboration.vote_threshold", 0.5, minimum=0.0,
        )
        pool = [w for w in dict.fromkeys(workers or self.pick()) if w in self.profiles]
        if not pool:
            return {"ok": False, "consensus": False, "report": "错误：没有可投票的子智能体。"}

        # 显式 timeout = 总预算：两阶段各分一半（None 时各自自适应，不折半）。
        # 收集阶段产出长方案、投票阶段只回一个数字，量级不对称，但对半分是
        # "总时长有上界"的最简单契约，避免调用方误判 vote 收工时间。
        stage_timeout = timeout / 2.0 if timeout is not None else None

        # 第一步：收集各工人方案（失败者淘汰，不进候选池；顺序确定性）
        candidates = self.collect(pool, prompt, timeout=stage_timeout)
        if not candidates:
            return {"ok": False, "consensus": False, "report": "错误：所有工人均未给出可用方案，无法投票。"}

        # 第二步：编号后发给全体投票
        result = self._run_ballot(candidates, pool, stage_timeout)
        if result is None:
            return {"ok": False, "consensus": False, "report": "错误：投票结果无法解析（工人未输出编号）。"}
        best_idx, tally, total, best_n, invalid = result
        ratio = best_n / total
        body = ["## 候选方案", *[f"[{i}] {w}：{ans[:96]}…" for i, (w, ans) in enumerate(candidates, 1)]]
        body.append("## 投票结果")
        for i in range(1, len(candidates) + 1):
            body.append(f"方案[{i}]: {tally.get(i, 0)}/{total} 票")
        if invalid:
            body.append(
                f"（另有 {invalid} 张票投了不存在的编号，已忽略；比例按有效票 {total} 张计算）"
            )
        consensus = ratio >= th
        if consensus:
            winner = candidates[best_idx - 1]
            body.append(f"## 共识达成（{best_n}/{total} 票 ≥ {th:.0%}）")
            body.append(f"胜出方案（{winner[0]}）：\n{winner[1]}")
        else:
            body.append(f"## 未达共识（最高 {best_n}/{total}，阈值 {th:.0%}）")
            body.append("请主智能体结合候选内容自行裁决或要求重投。")
        return {"ok": True, "consensus": consensus, "report": "\n".join(body)}
