"""错误文案脱敏：原始异常文本会外流到 stdout 协议行与模型上下文，先抹掉凭证形态的串。

为什么要单独一层：ERR_OTHER、工具兜底、协议兜底这三处只能拿到 `str(e)`，而底层
（httpx / openai SDK）的异常串经常回显请求 URL——把 key 放在 query 里的中转端点就会
在这里泄密；bridge/MCP 的 stdout 是纯 JSON 行协议，CLI 又会把这些行交给外部主，
其中一部分还会被拼进发给工人的提示词，等于把凭证送进第三方上下文。
其余错误分支用的是固定文案，只有兜底分支必须透传异常原文，故只在这里过一遍。
"""

from __future__ import annotations

import re

from .constants import SECRET_NAME_TAILS

# URL 的 query 段：凭证最常藏在这里（?key=… / ?api_key=…）
_URL_QUERY = re.compile(r"(?:https?|wss?)://[^\s'\"<>]+?\?[^\s'\"<>]*")
# URL 的 userinfo 段：`https://user:pass@host/v1` 把凭证放在**路径前面**，上面的 query
# 规则要求有 `?` 才命中，无 query 的端点串会整段原样外流（异常文案里回显请求 URL 正是
# 这条通道的主要来源）。只抹 `scheme://` 到第一个 `@` 之间的部分，host/path 保留——
# 排障要看的恰恰是 host。没有 `@` 的正常 URL 不命中（字符类里排除了 `/`）。
_URL_USERINFO = re.compile(r"((?:https?|wss?)://)[^\s'\"<>/?@]+@([^\s'\"<>/@]+)")
# 成对的「变量名 = 值」形态：名字必须以密钥词结尾（KEY/TOKEN/SECRET/PASSWORD/…），
# 且只在该词是**独立一段**时才算密钥 —— 旧写法用 `\b(key|token|...)=` 匹配，
# `NODE_A_KEY=sk-xxx`、`CF_API_TOKEN=yyy` 这类下划线后缀名一个都不命中（`_` 与 `K`
# 之间没有词边界），而本项目的池 key 恰恰全是这种命名。`MONKEY=3` 不该被误抹。
# 值本身若形如 sk-/Bearer，即使名字没命中也会被上面的规则抹掉（双重兜底）。
_SECRET_KV = re.compile(
    r"(?i)\b([A-Za-z0-9_\-]*(?:api[ _\-]?key|key|token|secret|password|passwd|credential))"
    r"(\s*[=:]\s*)([^\s&'\",;]+)"
)


def _is_secret_name(name: str) -> bool:
    """分段命名（`NODE_A_KEY` / `CF_API_TOKEN`）的末段是否密钥词。

    只按末段判、不按整名后缀判：`MONKEY=3` 这种同尾词若也抹，就把正常正文里的
    KV 抹成****（对外文案的口径是"宁可少抹也不误伤"）；紧凑写法（`MYKEY=` /
    `GHTOKEN=`）在异常文案里几乎不出现，真出现时值本身通常形如 sk-/Bearer，
    会被上面两条规则抹掉。子进程 env 通道另有更保守的判定（builtin._child_env
    同时看末段与后缀），因为那条路"误剔一个无关变量"的代价远小于"抄走凭据"。
    """
    up = name.upper().replace("-", "_").replace(" ", "_")
    return up.rsplit("_", 1)[-1] in SECRET_NAME_TAILS


def _mask_secret_kv(m: re.Match) -> str:
    """命中"名字是分段密钥命名"的 KV 才抹值，否则原样返回。"""
    if _is_secret_name(m.group(1)):
        return f"{m.group(1)}{m.group(2)}****"
    return m.group(0)


_SECRET_PATTERNS: list[tuple[re.Pattern, object]] = [
    (_URL_QUERY, lambda m: m.group(0).split("?", 1)[0] + "?<query 已隐去>"),
    (_URL_USERINFO, lambda m: m.group(1) + "****@" + m.group(2)),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{6,}"), lambda m: "sk-****"),
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._\-]{6,}"), lambda m: m.group(1) + "****"),
    (_SECRET_KV, _mask_secret_kv),
]
_LIMIT = 200


def mask_secrets(value: object) -> str:
    """只做凭证掩码、**不截断**：给工具输出（run_shell 的 stdout/stderr）这类需要保留
    完整正文的出口用。

    为什么不能直接复用 safe_error_text：它按错误文案的口径截到 200 字符，用它抹命令输出
    等于把 8000 字符的 stdout 砍成两行——排障信息全丢。掩码规则两边共用同一份
    `_SECRET_PATTERNS`，只是截不截断不同。
    """
    text = str(value or "")
    for pat, repl in _SECRET_PATTERNS:
        text = pat.sub(repl, text)
    return text


def safe_error_text(value: object, limit: int = _LIMIT) -> str:
    """把任意异常/文本压成"短、不含凭证"的错误文案（用于回显给外部主与模型）。"""
    text = mask_secrets(value).strip()
    if not text:
        return "未知错误"
    return text if len(text) <= limit else text[:limit] + "…（已截断）"
