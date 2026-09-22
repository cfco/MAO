"""CLI 入口。

  python -m agent bridge                 外部智能体驱动模式（谁启动驱动，谁当主智能体）
  python -m agent chat                   交互对话（池里多个智能体时交互选主）
  python -m agent chat --orchestrator 名  直接指定主智能体
  python -m agent chat --solo            不启用工人协作
  python -m agent run "任务" [--orchestrator 名] [--solo] [--stream]  --stream 时事件以 JSON 行输出
  python -m agent pipeline "任务" [--draft a,b --review c --revise a]  固定流水线（起草→评审→修订择优）
  python -m agent web                    启动 Web 服务
"""
from __future__ import annotations

import argparse
import os
import sys

from .bridge import serve as bridge_serve
from .config import ROOT, Config, load_config
from .core.agent import Agent
from .core.orchestrator import WorkerPool
from .core.pipeline import Pipeline


def _print_event(event: dict) -> None:
    t = event.get("type")
    if t == "tool_start":
        print(f"\n  [工具] {event['name']} ...")
    elif t == "tool_result":
        out = str(event.get("result", ""))
        preview = out if len(out) <= 600 else out[:600] + " ...[已截断]"
        print(f"  [结果] {preview}")
    elif t == "pipeline_stage":
        # CLI pipeline 只发 pipeline_stage 事件；不处理的话整个流水线阶段全程静默，
        # 直到最终结果才有一次输出。
        print(f"  [阶段] {event.get('stage')}: {event.get('summary', '')}")


def _print_json_event(event: dict) -> None:
    """输出统一事件 schema 的 JSON 行（与 bridge/Web 一致），供脚本解析复用。"""
    import json

    from .core.events import to_event

    print(json.dumps(to_event(event), ensure_ascii=False))


def pick_orchestrator(cfg: Config, preferred: str | None, interactive: bool, stream: bool = False):
    """确定主智能体。返回 (AgentProfile | None)。池为空时用兜底单模型。

    stream=True（run --stream）时所有诊断信息改走 stderr：此时契约是
    "stdout 只放 JSON 事件行"，混入人类可读提示会让按行解析的脚本直接报错。
    """
    def log(msg: str = "") -> None:
        print(msg, file=sys.stderr if stream else sys.stdout)

    if cfg.agent_profiles:
        if preferred:
            p = cfg.profile(preferred)
            if p is None:
                names = ", ".join(p.name for p in cfg.agent_profiles)
                log(f"错误：智能体 '{preferred}' 不在池中。可用：{names}")
                sys.exit(1)
            log(f"[主智能体] {p.name} ({p.model})")
            return p
        if len(cfg.agent_profiles) == 1 or not interactive:
            p = cfg.agent_profiles[0]
            log(f"[主智能体] {p.name} ({p.model})")
            return p
        log("选择主智能体（其余自动成为工人）：")
        for i, p in enumerate(cfg.agent_profiles, 1):
            note = f"  {p.note}" if p.note else ""
            log(f"  {i}. {p.name}  [{p.model}]{note}")
        while True:
            try:
                raw = input("输入编号或名称: ").strip()
            except (EOFError, KeyboardInterrupt):
                sys.exit(1)
            if raw.isdigit() and 1 <= int(raw) <= len(cfg.agent_profiles):
                return cfg.agent_profiles[int(raw) - 1]
            p = cfg.profile(raw)
            if p:
                return p
            log("无效输入，请重试。")
    # 池空 → 兜底单模型
    llm = cfg.llm
    base = str(llm.get("base_url", ""))
    if not str(llm.get("api_key") or "").strip() and "localhost" not in base and "127.0.0.1" not in base:
        log("未配置任何智能体。两种方式：")
        log("  1. 在 config.yaml 的 agents 下添加智能体池（推荐，可多模型协作）")
        log('  2. 配置兜底单模型：setx LLM_API_KEY "你的key" 后重开终端')
        sys.exit(1)
    return None


def build_agent(cfg: Config, args, interactive: bool = False, profile=None) -> Agent:
    """构造主智能体。

    profile 显式传入时直接沿用、不再走选主流程 —— /new 开新会话时用它保持当前主。
    否则 interactive=False 会让 pick_orchestrator 直接取池内第一个，用户先前交互
    选中的主会被悄悄换掉（CLI 与顶栏都没有任何提示）。
    """
    # --stream 时 stdout 只承载 JSON 事件行，启动期的诊断信息统一走 stderr
    stream = bool(getattr(args, "stream", False))
    err = sys.stderr if stream else sys.stdout
    if profile is not None:
        print(f"[主智能体] {profile.name} ({profile.model})（沿用当前主）", file=err)
    else:
        profile = pick_orchestrator(
            cfg, getattr(args, "orchestrator", None), interactive, stream=stream
        )
    bot = Agent(cfg, profile=profile, enable_workers=not getattr(args, "solo", False))
    for line in bot.mcp_status:
        print(line, file=err)
    # 启动时清一次历史会话（只留最近 30 个），防 data/sessions 无限膨胀
    from .core.session import Session
    removed = Session.cleanup()
    if removed:
        print(f"[清理] 已清理 {removed} 个过期会话文件", file=err)
    return bot


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent", description="自研多智能体协作系统")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("bridge", help="外部智能体驱动模式（谁启动驱动，谁当主智能体）")

    p_chat = sub.add_parser("chat", help="交互对话")
    p_chat.add_argument("--orchestrator", help="直接指定主智能体名")
    p_chat.add_argument("--solo", action="store_true", help="不启用工人协作")
    p_chat.add_argument("--session-flush-batch", type=int, default=None,
                        help="覆盖会话落盘限流的批量阈值（条数），默认读 config.yaml")
    p_chat.add_argument("--session-flush-interval", type=float, default=None,
                        help="覆盖会话落盘限流的时间阈值（秒），默认读 config.yaml")

    p_run = sub.add_parser("run", help="执行一次性任务")
    p_run.add_argument("task", help="任务描述")
    p_run.add_argument("--orchestrator", help="直接指定主智能体名")
    p_run.add_argument("--solo", action="store_true", help="不启用工人协作")
    p_run.add_argument("--stream", action="store_true",
                       help="过程事件以 JSON 行输出（与 bridge 协议一致，便于脚本解析）")
    p_run.add_argument("--session-flush-batch", type=int, default=None,
                       help="覆盖会话落盘限流的批量阈值（条数），默认读 config.yaml")
    p_run.add_argument("--session-flush-interval", type=float, default=None,
                       help="覆盖会话落盘限流的时间阈值（秒），默认读 config.yaml")

    p_pipe = sub.add_parser("pipeline", help="固定流水线：起草→评审→修订择优（需要多个智能体）")
    p_pipe.add_argument("task", help="要打磨产出的任务描述（方案/报告/代码/文案等）")
    p_pipe.add_argument("--draft", help="起草工人名，逗号分隔；缺省池内全部")
    p_pipe.add_argument("--review", help="评审工人名，逗号分隔；缺省池内全部")
    p_pipe.add_argument("--revise", help="修订工人名，逗号分隔；缺省用起草工人")

    p_web = sub.add_parser("web", help="启动 Web 服务")
    p_web.add_argument("--session-flush-batch", type=int, default=None,
                       help="覆盖会话落盘限流的批量阈值（条数），默认读 config.yaml")
    p_web.add_argument("--session-flush-interval", type=float, default=None,
                       help="覆盖会话落盘限流的时间阈值（秒），默认读 config.yaml")

    args = parser.parse_args()

    # 命令行覆盖会话落盘限流参数：写入进程内环境变量，load_config 会优先采用，
    # 对同进程内的 chat/run/pipeline/web/bridge 各入口（含 web 的 runpy 与 bridge）均生效。
    if getattr(args, "session_flush_batch", None) is not None:
        os.environ["MAO_SESSION_FLUSH_BATCH"] = str(args.session_flush_batch)
    if getattr(args, "session_flush_interval", None) is not None:
        os.environ["MAO_SESSION_FLUSH_INTERVAL"] = str(args.session_flush_interval)

    # bridge / run --stream：先归一 stdio 为 UTF-8，再 load_config。
    # 两者的契约都是 stdout 纯 JSON 行（含中文），Windows 重定向流默认 GBK：
    # 不归一则外部按 UTF-8 解码直接报错（实测 run --stream 管道输出 0xC4 起头的 GBK 字节）。
    # 配置缺失变量的 [配置警告] 在 load_config 阶段就会打到 stderr，也必须晚于此处。
    if args.cmd == "bridge" or (args.cmd == "run" and getattr(args, "stream", False)):
        from .bridge import reconfigure_streams
        reconfigure_streams()

    cfg = load_config()

    if args.cmd == "bridge":
        bridge_serve(cfg)
        return

    if args.cmd == "web":
        import runpy
        runpy.run_path(str(ROOT / "web" / "server.py"), run_name="__main__")
        return

    if args.cmd == "pipeline":
        # 固定流水线不依赖主智能体（不选主），直接调度池内全部模型当工人
        pool = WorkerPool(cfg)
        if not pool.names():
            print("错误：智能体池为空，无法执行流水线。请先在 config.yaml 配置多个智能体。")
            sys.exit(1)
        pipe = Pipeline(pool)
        # 打印缺省参与名单（受 collaboration.max_participants 约束）：显式传 --draft/--review
        # 时会覆盖各自阶段，这里给出的是"不指定时"实际会用到的工人。
        default_participants = pool.pick()
        print(
            f"固定流水线 起草→评审→修订 开始"
            f"（缺省参与 {len(default_participants)} 名：{', '.join(default_participants)}）"
        )
        result = pipe.run(
            args.task,
            draft_workers=[w for w in args.draft.split(",") if w.strip()] if args.draft else None,
            review_workers=[w for w in args.review.split(",") if w.strip()] if args.review else None,
            revise_workers=[w for w in args.revise.split(",") if w.strip()] if args.revise else None,
            on_event=_print_event,
        )
        print(result)
        return

    bot = build_agent(cfg, args, interactive=(args.cmd == "chat"))

    if args.cmd == "run":
        handler = _print_json_event if args.stream else _print_event
        try:
            result = bot.run(args.task, on_event=handler)
            # 与 bridge 一致：正式结果作为最后一行 JSON 输出
            if args.stream:
                import json
                print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
            else:
                print(result)
        finally:
            bot.close()
        return

    mode = "solo" if args.solo else ("swarm·多智能体协作" if bot.worker_pool else "solo")
    print(f"交互模式 [{mode}]：输入内容开始对话，/new 开新会话，/exit 退出。")
    while True:
        try:
            user = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            bot.close()
            return
        if not user:
            continue
        if user == "/exit":
            print("再见")
            bot.close()
            return
        if user == "/new":
            keep_profile = bot.profile  # 保住当前主，别被池内第一个顶掉
            bot.close()
            bot = build_agent(cfg, args, interactive=False, profile=keep_profile)
            print("已开启新会话。")
            continue
        print("助手> ", end="", flush=True)
        # bot.run 内部已对 LLMError/会话异常做了友好兜底；这里再兜一层，
        # 防止意外异常（如 OOM、底层库裸异常）让交互模式直接崩掉丢会话。
        try:
            print(bot.run(user, on_event=_print_event))
        except Exception as e:  # noqa: BLE001
            print(f"\n[本轮出错] {type(e).__name__}: {e}（会话已保留，可继续输入）")


if __name__ == "__main__":
    main()
