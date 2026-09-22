"""工具基类与注册表：内置工具、MCP 工具、Skill 工具统一适配到同一接口。"""
from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any


class Tool:
    """统一工具接口。所有工具来源都适配成：name + description + JSON Schema + execute。"""

    name: str = ""
    description: str = ""
    input_schema: dict = {}

    def execute(self, args: dict, ctx: Any | None = None) -> str:
        raise NotImplementedError

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }


class FunctionTool(Tool):
    """用 Python 函数直接构造一个工具。

    `func` 可写成单参 `func(args)`（绝大多数工具），也可写成双参 `func(args, ctx)`
    ——双参版本用于需要「边执行边冒进度事件」或「响应用户取消」的工具（如 ask_workers）。
    是否传 ctx 由本类按签名自动判定：单参工具即便注册表传了 ctx 也不会收到（保持向后
    兼容，bridge call_tool / 测试等无 ctx 的调用路径照常工作）。
    """

    def __init__(self, name: str, description: str, input_schema: dict, func: Callable[..., str]):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._func = func
        self._wants_ctx = _accepts_ctx(func)

    def execute(self, args: dict, ctx: Any | None = None) -> str:
        if self._wants_ctx:
            return self._func(args, ctx)
        return self._func(args)


def _accepts_ctx(func: Callable[..., Any]) -> bool:
    """该工具函数是否需要接收第二个参数 ctx。"""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    positional = [
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
    return has_varargs or len(positional) >= 2


class ToolRegistry:
    """工具注册表：Agent 主循环只和它打交道，不关心工具来自哪里。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具名重复: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def restrict(self, allowed: list[str]) -> list[str]:
        """会话级工具隔离：只保留白名单内的工具，返回被裁剪掉的工具名。

        用于多会话/多租户场景，给不同会话不同的工具权限。
        """
        allowed_set = set(allowed)
        removed = [n for n in self._tools if n not in allowed_set]
        for n in removed:
            del self._tools[n]
        return removed

    def info_list(self) -> list[dict]:
        """列出全部工具的 (name, description)，供对外展示（bridge list_tools 等）。"""
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]

    def openai_schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: str | dict, ctx: Any | None = None) -> str:
        """执行一次工具调用，永不抛异常，错误以文本返回给模型自行处理。

        ctx 可选地携带 {"emit": fn, "cancel_event": Event}，透传给声明了双参签名的
        工具（见 FunctionTool）。无 ctx 的调用路径（bridge call_tool、单测）照常工作。
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"错误：未知工具 '{name}'，可用工具：{', '.join(self._tools)}"
        try:
            if isinstance(arguments, str):
                args = json.loads(arguments) if arguments.strip() else {}
            else:
                args = arguments
        except json.JSONDecodeError as e:
            return f"错误：工具参数不是合法 JSON：{e}"
        try:
            return tool.execute(args, ctx)
        except Exception as e:  # noqa: BLE001 - 工具错误必须转成文本回给模型
            return f"工具执行出错：{type(e).__name__}: {e}"
