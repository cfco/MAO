"""错误文案脱敏：原始异常文本会外流到 stdout 协议行与模型上下文，先抹掉凭证形态的串。

为什么要单独一层：ERR_OTHER、工具兜底、协议兜底这三处只能拿到 `str(e)`，而底层
（httpx / openai SDK）的异常串经常回显请求 URL——把 key 放在 query 里的中转端点就会
在这里泄密；bridge/MCP 的 stdout 是纯 JSON 行协议，CLI 又会把这些行交给外部主，
其中一部分还会被拼进发给工人的提示词，等于把凭证送进第三方上下文。
其余错误分支用的是固定文案，只有兜底分支必须透传异常原文，故只在这里过一遍。
"""

from __future__ import annotations

import re

# URL 的 query 段：凭证最常藏在这里（?key=… / ?api_key=…）
_URL_QUERY = re.compile(r"(?:https?|wss?)://[^\s'\"<>]+?\?[^\s'\"<>]*")
# 密钥形态：OpenAI 风格 sk-…、Authorization: Bearer …、以及 key/token=… 参数
# （替换值统一用函数，避免组引用与字面量两种写法混用）
_SECRET_PATTERNS: list[tuple[re.Pattern, object]] = [
    (_URL_QUERY, lambda m: m.group(0).split("?", 1)[0] + "?<query 已隐去>"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{6,}"), lambda m: "sk-****"),
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._\-]{6,}"), lambda m: m.group(1) + "****"),
    (
        re.compile(r"(?i)\b((?:api[_-]?key|secret|token|password|key)\s*[=:]\s*)[^\s&'\",;]+"),
        lambda m: m.group(1) + "****",
    ),
]
_LIMIT = 200


def safe_error_text(value: object, limit: int = _LIMIT) -> str:
    """把任意异常/文本压成"短、不含凭证"的错误文案（用于回显给外部主与模型）。"""
    text = str(value or "").strip()
    if not text:
        return "未知错误"
    for pat, repl in _SECRET_PATTERNS:
        text = pat.sub(repl, text)
    return text if len(text) <= limit else text[:limit] + "…（已截断）"
