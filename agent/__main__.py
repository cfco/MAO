"""CLI 入口：只保留 bridge 一条使用路线。

  python -m agent bridge    外部智能体驱动模式（谁启动驱动，谁当主智能体）

外部主 AI（千问办公 / WorkBuddy 等）通过 bridge 的 stdin/stdout JSON 行协议，
借用本项目的「手」（本机工具 / MCP / Skill）与「子 AI」（池内免费模型，经
ask / ask_many / ask_vote / run_pipeline 做并行与多重验证）。

本项目不再自带「选主自己跑」的入口：chat / run / pipeline / web 已移除——
主智能体统一由外部 AI 担任，MAO 只做被调用的执行器与工人池。
"""
from __future__ import annotations

import argparse
import os

from .bridge import reconfigure_streams
from .bridge import serve as bridge_serve
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agent", description="自研多智能体协作系统（外部主 AI 驱动，仅提供执行器与工人池）"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_bridge = sub.add_parser("bridge", help="外部智能体驱动模式（谁启动驱动，谁当主智能体）")
    p_bridge.add_argument("--session-flush-batch", type=int, default=None,
                          help="覆盖会话落盘限流的批量阈值（条数），默认读 config.yaml")
    p_bridge.add_argument("--session-flush-interval", type=float, default=None,
                          help="覆盖会话落盘限流的时间阈值（秒），默认读 config.yaml")

    args = parser.parse_args()

    # 命令行覆盖会话落盘限流参数：写入进程内环境变量，load_config 会优先采用。
    if getattr(args, "session_flush_batch", None) is not None:
        os.environ["MAO_SESSION_FLUSH_BATCH"] = str(args.session_flush_batch)
    if getattr(args, "session_flush_interval", None) is not None:
        os.environ["MAO_SESSION_FLUSH_INTERVAL"] = str(args.session_flush_interval)

    # bridge 契约：stdout 只放 JSON 行（含中文）。Windows 重定向流默认 GBK，
    # 必须先把 stdio 归一为 UTF-8，再 load_config —— 缺失变量的告警也要晚于此归一，
    # 否则会按 GBK 写入、外部按 UTF-8 解码直接报错。
    reconfigure_streams()
    bridge_serve(load_config())


if __name__ == "__main__":
    main()
