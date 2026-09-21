"""固定流水线：起草 → 评审 → 修订择优（M5：固定流水线策略）。

区别于 swarm（主智能体自主决定何时派工）：
- 流水线是「预设好的固定流程」，一次完整跑完，输出最终交付物。
- 适用场景：产出要求较高的作品（方案/报告/代码/文案），自动多轮打磨。

流程：
1. 起草（draft）：可选 worker 并行各出一稿，合并成候选草稿。
2. 评审（review）：把候选草稿交给评审工人，收集问题清单与改进建议。
3. 修订择优（revise）：指定工人各自综合评审意见出终稿，
   多份终稿时由工人投票择优，只返回胜出的那份（不再全量拼接）。

起草/评审阶段默认用池内全部工人（可分别指定）。
"""
from __future__ import annotations

from collections.abc import Callable

from ..tools.base import FunctionTool, ToolRegistry
from .orchestrator import WorkerPool

# 各阶段的角色设定（想让评审更严/更专业可以自己改）
DRAFT_SYSTEM = (
    "你是团队的起草者。严格按任务要求，直接产出完整、结构清晰、可交付的一稿。"
    "宁可长一点、具体一点，不要惜墨。不提问、不客套。"
)
REVIEW_SYSTEM = (
    "你是团队的评审专家。你只负责挑毛病找改进点，不负责重写。"
    "逐条列出：1) 事实/逻辑错误；2) 遗漏与不足；3) 具体可执行的改进建议。"
    "用编号列表输出。如果质量确实过关，也要明确说清楚哪里过关。"
)
REVISE_SYSTEM = (
    "你是团队的修订者。根据评审意见对初稿重新打磨，输出修订后的完整终稿。"
    "终稿必须自包含、可直接使用，不要附带解释说明文字。"
)

EventCallback = Callable[[dict], None]


class Pipeline:
    def __init__(self, pool: WorkerPool):
        self.pool = pool

    def _workers(self, spec: list[str] | None) -> list[str] | None:
        """把用户指定的工人名过滤为池内有效子集；None 表示用池内全部。"""
        if not spec:
            return None
        valid = [w for w in dict.fromkeys(spec) if w in self.pool.profiles]
        return valid or None

    def run(
        self,
        task: str,
        draft_workers: list[str] | None = None,
        review_workers: list[str] | None = None,
        revise_workers: list[str] | None = None,
        on_event: EventCallback | None = None,
    ) -> str:
        """跑一遍完整流水线，返回最终修订稿文本。

        on_event 事件：pipeline_stage{stage, summary} 每阶段一条。
        """
        def emit(stage: str, summary: str) -> None:
            if on_event:
                on_event({"type": "pipeline_stage", "stage": stage, "summary": summary})

        if not self.pool.names():
            return "错误：工人池为空，无法执行流水线。请先在 config.yaml 配置多个智能体。"

        # ---- 1. 起草：并行收集候选稿 ----
        draft_names = self._workers(draft_workers) or self.pool.names()
        emit("draft", f"起草：{len(draft_names)} 名工人并行出稿中…（{', '.join(draft_names)}）")
        draft_text = self.pool.ask_many(
            draft_names,
            f"【任务】\n{task}\n\n请作为起草者，直接输出这份任务的完整初稿（方案/代码/文案等）。",
            DRAFT_SYSTEM,
        )

        # ---- 2. 评审：把全部候选稿交给评审工人 ----
        review_names = self._workers(review_workers) or self.pool.names()
        emit("review", f"评审：{len(review_names)} 名工人审查草稿并给意见…")
        review_text = self.pool.ask_many(
            review_names,
            f"【任务】\n{task}\n\n【以下为候选初稿】\n{draft_text}\n\n"
            f"请作为评审专家，逐条指出问题与改进建议（编号列表）。",
            REVIEW_SYSTEM,
        )

        # ---- 3. 修订择优：修订工人各出一份终稿，投票取最优的那份 ----
        revise_names = self._workers(revise_workers) or draft_names
        emit("revise", f"修订：{len(revise_names)} 名工人综合评审意见出终稿…")
        finals = self.pool.collect(
            revise_names,
            f"【任务】\n{task}\n\n【初稿】\n{draft_text}\n\n【评审意见】\n{review_text}\n\n"
            f"请作为修订者，综合评审意见输出完整终稿。",
            REVISE_SYSTEM,
        )
        if not finals:
            return "错误：修订阶段所有工人都未能产出终稿，流水线中止。"
        if len(finals) == 1:
            winner_w, final_text = finals[0]
            selection = "仅一名工人产出终稿，直接采用"
        else:
            picked = self.pool.select_best(finals, voters=revise_names)
            if picked is not None:
                winner_w, final_text, selection = picked
            else:
                # 投票无法解析（工人都没输出编号）：按确定性顺序取第一份，不回退全量拼接
                winner_w, final_text = finals[0]
                selection = "投票未能解析，按工人顺序取第一份"
        emit("revise", f"修订完成：采用 {winner_w} 的终稿（{selection}）")

        emit("done", "流水线完成，三个臭皮匠已顶一个诸葛亮。")
        # 最终输出精简：初稿和评审只留摘要，胜出终稿给全文（核心交付物）。
        # 终稿不再多份全量拼接——旧实现把每个修订工人的终稿都塞回来，冗长且无结论。
        _DRAFT_PREVIEW = 2000   # 初稿摘要保留字符数
        _REVIEW_PREVIEW = 2000  # 评审摘要保留字符数
        draft_short = draft_text.strip()[:_DRAFT_PREVIEW]
        if len(draft_text.strip()) > _DRAFT_PREVIEW:
            draft_short += f"\n...[初稿已截断，完整长度 {len(draft_text)} 字]"
        review_short = review_text.strip()[:_REVIEW_PREVIEW]
        if len(review_text.strip()) > _REVIEW_PREVIEW:
            review_short += f"\n...[评审已截断，完整长度 {len(review_text)} 字]"
        return (
            f"## 流水线结果（起草 → 评审 → 修订择优）\n\n"
            f"### 初稿摘要（{len(draft_names)} 名起草工人）\n{draft_short}\n\n"
            f"### 评审意见摘要（{len(review_names)} 名评审工人）\n{review_short}\n\n"
            f"### 修订终稿（完整 · {winner_w} · {selection}）\n{final_text.strip()}"
        )


def register_pipeline_tool(registry: ToolRegistry, pipeline_factory) -> None:
    """把流水线注册为主智能体的工具（主自主决定何时走固定流程）。"""

    def run(args: dict) -> str:
        pl = pipeline_factory()
        return pl.run(
            str(args.get("task", "")).strip(),
            draft_workers=[str(w) for w in (args.get("draft_workers") or [])] or None,
            review_workers=[str(w) for w in (args.get("review_workers") or [])] or None,
            revise_workers=[str(w) for w in (args.get("revise_workers") or [])] or None,
        )

    registry.register(FunctionTool(
        name="run_pipeline",
        description=(
            "跑一遍固定流水线「起草→评审→修订择优」：多名子智能体并行起草，另一批评审挑错，"
            "再让修订工人各出终稿并投票择优，只返回胜出的那份终稿。"
            "适合方案、报告、代码、文案等需要反复打磨的高质量产出。"
            "返回包含初稿摘要/评审意见摘要/胜出终稿的结果。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "要产出的任务，写清要求、格式、验收标准"},
                "draft_workers": {"type": "array", "items": {"type": "string"}, "description": "起草工人名单，缺省全部"},
                "review_workers": {"type": "array", "items": {"type": "string"}, "description": "评审工人名单，缺省全部"},
                "revise_workers": {"type": "array", "items": {"type": "string"}, "description": "修订工人名单，缺省用起草工人"},
            },
            "required": ["task"],
        },
        func=run,
    ))
