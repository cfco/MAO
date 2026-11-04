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
from .core.redact import safe_error_text


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agent", description="自研多智能体协作系统（外部主 AI 驱动，仅提供执行器与工人池）"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    sub.add_parser("bridge", help="外部智能体驱动模式（stdin/stdout JSON 行长驻进程）")
    sub.add_parser(
        "mcp", help="以标准 MCP server（stdio）接入宿主：Claude Desktop / Cursor / Qoder 等"
    )

    p_call = sub.add_parser("call", help="单发一条指令即退出（脚本接入用）")
    p_call.add_argument(
        "cmd",
        help="指令名：ping/list_agents/health/latency/set_economy/refresh_nodes/"
        "list_tools/call_tool/load_skill/run_skill_script/"
        "ask/ask_many/ask_vote/run_review",
    )
    p_call.add_argument(
        "payload",
        nargs="?",
        default="",
        help='JSON 参数（省略则从 stdin 读一行），如 \'{"prompt":"...","agent":"x"}\'',
    )
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
        # 这一行同样进 stdout 的 JSON 协议：与 bridge 一个口径，先过脱敏
        print(
            json.dumps(
                {"ok": False, "error": f"参数 JSON 解析失败: {safe_error_text(e)}"},
                ensure_ascii=False,
            )
        )
        sys.exit(2)
    payload["cmd"] = args.cmd

    from .bridge import Bridge

    # 构造（load_config / Bridge）也在 try 内（2026-11-04 审计 P3）：call 的契约是
    # "一次调用一行回包"，构造抛出去 stdout 只剩 stderr 栈，按行解析 JSON 的调用方
    # 拿到空行——构造失败同样要回一行 ok:false。
    try:
        bridge = Bridge(load_config())
        try:
            resp = bridge.handle(payload)
        except Exception as e:  # noqa: BLE001 - 单发模式同样必须回一行 JSON，不能只留崩栈
            # bridge 的长驻循环对每条指令都有兜底；call 是"一次调用一行回包"的契约，
            # 抛出去就只剩 stderr 的栈，脚本按 stdout 解析 JSON 的调用方会拿到空行。
            resp = {"ok": False, "error": f"{type(e).__name__}: {safe_error_text(e)}"}
    except Exception as e:  # noqa: BLE001 - 构造失败也要守住一行 JSON 的契约
        resp = {"ok": False, "error": f"初始化失败：{type(e).__name__}: {safe_error_text(e)}"}
    finally:
        bridge_close = locals().get("bridge")
        if bridge_close is not None:
            bridge_close.close()
    print(json.dumps(resp, ensure_ascii=False))
    sys.exit(0 if resp.get("ok") else 1)


if __name__ == "__main__":
    main()
