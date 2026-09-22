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


class ToolResult(str):
    """工具返回值：默认（裸 str）即成功；需要显式判失败时返回 `ToolResult(text, ok=False)`。

    设计成 str 子类，是为了让 `execute()` 的文本契约、以及"输出即任意文本"的工具
    （read_file/run_shell/list_dir/MCP 透传）不再被"错误：…"前缀启发式误判（#3）：
    - 文件内容/命令输出恰好以"错误："开头 → 裸 str → ok=True（不是工具失败）；
    - 路径越界、命令被拦、超时等**真·工具错误** → ToolResult(..., ok=False)。
    既有对返回文本做 `==` / `in` / 切片 / startswith 的调用与测试全部照旧可用。
    """

    def __new__(cls, text: str, ok: bool = True) -> ToolResult:
        obj = super().__new__(cls, text)
        obj.ok = ok
        return obj


class FunctionTool(Tool):
    """用 Python 函数直接构造一个工具。

    `func` 可写成单参 `func(args)`（当前全部内置/Skill 工具），也可写成双参
    `func(args, ctx)`——双参是留给"边执行边冒进度事件 / 响应用户取消"这类流式工具的
    扩展缝（当前无内置工具用到，但注册表会按签名自动决定是否透传 ctx）。单参工具即便
    注册表传了 ctx 也不会收到（bridge call_tool / 测试等无 ctx 的调用路径照常工作）。
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

    def info_list(self) -> list[dict]:
        """列出全部工具的 (name, description)，供对外展示（bridge list_tools 等）。"""
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]

    def openai_schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def run(self, name: str, arguments: str | dict, ctx: Any | None = None) -> dict:
        """执行一次工具调用，返回结构化 {ok, result, error}（#3：让 ok 反映真实成败）。

        ok 判定（不再嗅探返回值文本前缀）：
        - 未知工具 / 参数非合法 JSON / 工具抛异常 → ok=False（注册表控制流错误）；
        - 工具返回 ToolResult → 取其 .ok；
        - 工具返回裸 str → 成功（ok=True），哪怕内容以"错误："开头（可能是文件/命令正文）。
        result 恒为文本；error 仅失败时给出（等于 result）。
        """
        tool = self._tools.get(name)
        if tool is None:
            msg = f"错误：未知工具 '{name}'，可用工具：{', '.join(self._tools)}"
            return {"ok": False, "result": msg, "error": msg}
        try:
            if isinstance(arguments, str):
                args = json.loads(arguments) if arguments.strip() else {}
            else:
                args = arguments
        except json.JSONDecodeError as e:
            msg = f"错误：工具参数不是合法 JSON：{e}"
            return {"ok": False, "result": msg, "error": msg}
        try:
            out = tool.execute(args, ctx)
        except Exception as e:  # noqa: BLE001 - 工具错误必须转成文本回给模型
            msg = f"工具执行出错：{type(e).__name__}: {e}"
            return {"ok": False, "result": msg, "error": msg}
        ok = bool(getattr(out, "ok", True))  # 裸 str → True；ToolResult → 自带标志
        text = str(out)
        return {"ok": ok, "result": text, "error": "" if ok else text}

    def execute(self, name: str, arguments: str | dict, ctx: Any | None = None) -> str:
        """执行一次工具调用，返回文本（永不抛异常）。run() 的文本包装，向后兼容。"""
        return self.run(name, arguments, ctx)["result"]
