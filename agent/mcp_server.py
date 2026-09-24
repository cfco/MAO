"""MAO 的 MCP server 入口：把 bridge 能力原样暴露成标准 MCP 工具。

为什么要有这一层：bridge 的裸 JSON 行协议要求每个接入方自己写子进程驱动
（握手、逐行读写、UTF-8 归一），接一次麻烦一次。而主流宿主（Claude Desktop /
Cursor / Qoder / 千问办公等）几乎都内置了 MCP client——本模块把 Bridge 的
13 条指令逐一映射为 MCP 工具，接入退化为在宿主配置里加一段
`{"command": "uv", "args": ["run", "mao", "mcp"]}`，协议细节全部由 MCP 消化。

实现取向：不复制任何业务逻辑，每个工具都只是把入参拼成 bridge 请求、原样调
`Bridge.handle()`——bridge 与 mcp 两条外壳共享同一个内核，行为（含错误文案、
无状态语义）逐字一致，测试也只需盯一处。
"""
from __future__ import annotations

from typing import Any

from . import __version__
from .bridge import Bridge
from .config import Config

_SERVER_INSTRUCTIONS = (
    "MAO：外部主 AI 的执行器 + 免费模型工人池。工具均无状态，历史上下文由调用方自持。"
    "用法建议：派工前先 health 预检（available=false 的别硬派），按 tags 决定哪类活派给谁；"
    "call_tool 使用本机工具（run_shell/文件读写/list_dir，范围=项目根+配置的 workspace）；"
    "ask/ask_many/ask_vote/run_review 调度池内免费子 AI（纯文本执行，无本地工具）。"
    "⚠ 安全：call_tool→run_shell 会在**运行本进程的这台机器**上执行真实命令，run_shell 是"
    "命令卫生黑名单、并非沙箱。接入前请确认你信任该宿主；生产环境建议用容器/独立账户隔离，"
    "并只把 workspace 配成必要的目录。"
)


def build_server(cfg: Config) -> tuple[Any, Bridge]:
    """构造 MCP server（MCPServer, stdio）与它背后的 Bridge 内核。

    返回 (app, bridge)：调用方负责在退出时 bridge.close() 释放连接池。
    惰性 import mcp：只用 bridge/call 的宿主不必加载 server 端依赖。
    """
    from mcp.server.mcpserver import MCPServer

    bridge = Bridge(cfg)
    app = MCPServer(name="MAO", version=__version__, instructions=_SERVER_INSTRUCTIONS)

    def _h(req: dict) -> dict:
        return bridge.handle(req)

    @app.tool(name="ping", description="探活：返回版本与池内智能体名")
    def ping() -> dict:
        return _h({"cmd": "ping"})

    @app.tool(name="list_agents", description="智能体池清单（含能力标签与上下文窗口，不含 key）")
    def list_agents() -> dict:
        return _h({"cmd": "list_agents"})

    @app.tool(name="health", description="派工前预检：各子 AI 可用性快照（冷却/当日隔离/标签/延迟ms）")
    def health() -> dict:
        return _h({"cmd": "health"})

    @app.tool(name="latency",
              description="延迟档案与探测状态（问题5）：各工人往返耗时中位数（毫秒）+ 探测调度；"
                          "probe=true 时先同步探测一轮**全部节点**（含不可用节点）再返回")
    def latency(probe: bool = False) -> dict:
        return _h({"cmd": "latency", "probe": probe})

    @app.tool(name="set_economy", description="省 token 开关（二元）：value 为 true/false，返回切换后的真实状态；非法值 ok:false 并保持原值")
    def set_economy(value: bool) -> dict:
        return _h({"cmd": "set_economy", "value": value})

    @app.tool(name="list_tools", description="可用本地工具清单（内置 + MCP + Skill）")
    def list_tools() -> dict:
        return _h({"cmd": "list_tools"})

    @app.tool(name="call_tool", description="执行一个本地工具（工具名以 list_tools 为准）")
    def call_tool(name: str, args: dict[str, Any] | None = None) -> dict:
        return _h({"cmd": "call_tool", "name": name, "args": args or {}})

    @app.tool(name="load_skill", description="读取技能全文（frontmatter + 正文）")
    def load_skill(name: str) -> dict:
        return _h({"cmd": "load_skill", "name": name})

    @app.tool(name="run_skill_script", description="执行技能目录 scripts/ 下的脚本")
    def run_skill_script(skill: str, script: str, args: dict[str, Any] | None = None) -> dict:
        return _h({"cmd": "run_skill_script", "skill": skill, "script": script, "args": args or {}})

    @app.tool(name="ask", description="把池内某个模型当纯文本子 AI 用；agent 留空走兜底单模型")
    def ask(prompt: str, agent: str = "", system: str = "") -> dict:
        return _h({"cmd": "ask", "prompt": prompt, "agent": agent, "system": system or None})

    @app.tool(name="ask_many",
              description="同一任务并行派多个工人（多方案对比/交叉验证）；"
                          "返回 workers 结构化逐工人结果（顺序与入参一致）+ results 文本；"
                          "workers 缺省取前 N 个可派工人（collaboration.max_participants）")
    def ask_many(prompt: str, workers: list[str] | None = None, system: str = "") -> dict:
        return _h({"cmd": "ask_many", "prompt": prompt, "workers": workers, "system": system or None})

    @app.tool(name="ask_vote",
              description="两步投票取共识：各出方案→对编号投票→超阈值给出胜出方案全文")
    def ask_vote(prompt: str, workers: list[str] | None = None, threshold: float | None = None) -> dict:
        return _h({"cmd": "ask_vote", "prompt": prompt, "workers": workers, "threshold": threshold})

    @app.tool(name="run_review",
              description="调用方给初稿（draft）+可选背景（context），多个工人只当评审团挑错；"
                          "返回结构化 reviews")
    def run_review(draft: str, context: str = "", workers: list[str] | None = None) -> dict:
        return _h({"cmd": "run_review", "draft": draft, "context": context, "workers": workers})

    return app, bridge


def serve(cfg: Config) -> None:
    """阻塞式 stdio MCP server：宿主连入即分发工具表，退出即释放资源。"""
    app, bridge = build_server(cfg)
    try:
        app.run("stdio")
    finally:
        bridge.close()
