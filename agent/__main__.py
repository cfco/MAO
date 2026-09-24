"""CLI 入口：bridge 一条内核，三种外壳（长驻协议 / MCP server / 单发调用）。

  python -m agent bridge          裸 JSON 行协议长驻进程（自研驱动方用）
  python -m agent mcp             标准 MCP server（stdio）：支持 MCP 的宿主填一段配置即接入
  python -m agent call <cmd> [json]  单发一条指令、打印一行响应即退出（脚本/技能包用）

外部主 AI（千问办公 / WorkBuddy / Claude Desktop 等）借用本项目的「手」（本机工具 /
MCP / Skill）与「子 AI」（池内免费模型，经 ask / ask_many / ask_vote / run_review
做并行派工与多重验证）。三者共享同一个 Bridge 内核，行为逐字一致。

本项目不再自带「选主自己跑」的入口：chat / run / pipeline / web 已移除——
主智能体统一由外部 AI 担任，MAO 只做被调用的执行器与工人池。
"""
from __future__ import annotations

import argparse
import json
import sys

from .bridge import reconfigure_streams
from .bridge import serve as bridge_serve
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agent", description="自研多智能体协作系统（外部主 AI 驱动，仅提供执行器与工人池）"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    sub.add_parser("bridge", help="外部智能体驱动模式（stdin/stdout JSON 行长驻进程）")
    sub.add_parser("mcp", help="以标准 MCP server（stdio）接入宿主：Claude Desktop / Cursor / Qoder 等")

    p_call = sub.add_parser("call", help="单发一条指令即退出（脚本接入用）")
    p_call.add_argument("cmd", help="指令名：ping/list_agents/health/latency/set_economy/"
                                     "list_tools/call_tool/load_skill/run_skill_script/"
                                     "ask/ask_many/ask_vote/run_review")
    p_call.add_argument("payload", nargs="?", default="",
                        help='JSON 参数（省略则从 stdin 读一行），如 \'{"prompt":"...","agent":"x"}\'')
    args = parser.parse_args()

    # 契约：stdout 只放 JSON 行（含中文）。Windows 重定向流默认 GBK，
    # 必须先把 stdio 归一为 UTF-8，再 load_config —— 缺失变量的告警也要晚于此归一，
    # 否则会按 GBK 写入、外部按 UTF-8 解码直接报错。
    reconfigure_streams()

    if args.mode == "bridge":
        bridge_serve(load_config())
        return

    if args.mode == "mcp":
        from .mcp_server import serve as mcp_serve
        mcp_serve(load_config())
        return

    # call：单发。payload 里带 cmd 也行（以位置参数为准），解析失败快速报错退出。
    text = args.payload.strip() or sys.stdin.readline().strip()
    try:
        payload = json.loads(text) if text else {}
        if not isinstance(payload, dict):
            raise ValueError("参数必须是 JSON 对象")
    except (json.JSONDecodeError, ValueError) as e:
        print(json.dumps({"ok": False, "error": f"参数 JSON 解析失败: {e}"}, ensure_ascii=False))
        sys.exit(2)
    payload["cmd"] = args.cmd

    from .bridge import Bridge
    bridge = Bridge(load_config())
    try:
        resp = bridge.handle(payload)
    finally:
        bridge.close()
    print(json.dumps(resp, ensure_ascii=False))
    sys.exit(0 if resp.get("ok") else 1)


if __name__ == "__main__":
    main()
