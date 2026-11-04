"""错误文案脱敏：原始异常文本会外流到 stdout 协议行与模型上下文，先抹掉凭证形态的串。

为什么要单独一层：ERR_OTHER、工具兜底、协议兜底这三处只能拿到 `str(e)`，而底层
（httpx / openai SDK）的异常串经常回显请求 URL——把 key 放在 query 里的中转端点就会
在这里泄密；bridge/MCP 的 stdout 是纯 JSON 行协议，CLI 又会把这些行交给外部主，
其中一部分还会被拼进发给工人的提示词，等于把凭证送进第三方上下文。
其余错误分支用的是固定文案，只有兜底分支必须透传异常原文，故只在这里过一遍。
"""

from __future__ import annotations

import re

from .constants import SECRET_NAME_SUBSTRINGS, SECRET_NAME_TAILS

# URL 的 query 段：凭证最常藏在这里（?key=… / ?api_key=…）
_URL_QUERY = re.compile(r"(?:https?|wss?)://[^\s'\"<>]+?\?[^\s'\"<>]*")
# URL 的 userinfo 段：`https://user:pass@host/v1` 把凭证放在**路径前面**，上面的 query
# 规则要求有 `?` 才命中，无 query 的端点串会整段原样外流（异常文案里回显请求 URL 正是
# 这条通道的主要来源）。只抹 `scheme://` 到第一个 `@` 之间的部分，host/path 保留——
# 排障要看的恰恰是 host。没有 `@` 的正常 URL 不命中（字符类里排除了 `/`）。
# 2026-11-04 审计：scheme 不再限定 http(s)/ws(s)——`postgres://u:p@h/db`、
# `redis://:p@h` 这类连接串里的密码同样会进错误文案，放行等于漏抹。
_URL_USERINFO = re.compile(r"([A-Za-z][A-Za-z0-9+.\-]*://)[^\s'\"<>/?@]+@([^\s'\"<>/@]+)")
# 成对的「变量名 = 值」形态：名字须含密钥词（KEY/TOKEN/SECRET/…，见 _is_secret_name），
# 且只在该词是**独立一段**或带常见后缀（KEY_ID/TOKEN_V2/CREDENTIALS）时才算密钥。
# 旧写法有两类漏网（2026-11-04 审计）：① 名字里密钥词后面还跟着后缀（`..._KEY_ID=`），
# 正则要求密钥词紧跟 `=`，一个都不命中；② Python dict / JSON repr 形态
# （`'NODE_A_KEY': 'xxx'`），名字与 `:` 之间隔着引号。两类都已在下面修正：
# 名字允许带后缀、分隔符允许夹一对引号。`MONKEY=3` 这类同形词现在也会被抹——
# 漏抹的代价是凭据进第三方上下文，误抹只是少看一段调试文本（宁严勿漏）。
_SECRET_KV = re.compile(
    r"(?i)\b([A-Za-z0-9_\-]*(?:api[ _\-]?key|key|token|secret|password|passwd|credential)[a-z0-9_\-]*)"
    r"(\s*['\"]?\s*[=:]\s*['\"]?)"
    r"([^\s&'\",;]+)"
)


def _is_secret_name(name: str) -> bool:
    """该变量名是否密钥形态（末段是密钥词，或密钥词出现在名字中段/带后缀）。

    两级判定（2026-11-04 审计从"只按末段"放宽）：
    1) 末段 ∈ SECRET_NAME_TAILS：`NODE_A_KEY` → `KEY`、`MYKEY` 紧凑式经后缀判定命中；
    2) 密钥词作为**子串**出现在名字里（SECRET_NAME_SUBSTRINGS）：兜住
       `AWS_ACCESS_KEY_ID` / `AUTH_TOKEN_V2` / `CREDENTIALS` 这类词不在末尾的命名。
    代价：MONKEY 这类同形词会被误抹。旧口径为"对外文案宁可少抹不误伤"只按末段判，
    但实测漏掉的恰是真实密钥命名（子进程 env 通道早就用了更严的口径），两道口子
    必须同严：漏抹一次就是把凭据送进第三方上下文，误抹只是少看一段调试文本。
    """
    up = name.upper().replace("-", "_").replace(" ", "_")
    if up.rsplit("_", 1)[-1] in SECRET_NAME_TAILS:
        return True
    return any(word in up for word in SECRET_NAME_SUBSTRINGS)


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
