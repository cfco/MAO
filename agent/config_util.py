"""配置解析的公共底座：路径口径、${VAR} 插值、模型清单拆分、类型容错告警。

为什么独立（2026-10-19）：`config.py` 一度 806 行，超仓库「单文件 ≤500 行」约定。
本模块只装**无状态小工具**，不依赖 `Config` / `AgentProfile`，因此能被加载流程
（`config_load.py`）与配置对象（`config.py`）同时 import，不形成环。

口径提醒：`ROOT` / `_free_nodes_path` / `_model_registry_path` 是全项目**唯一**的路径
真源，读侧（load_config）与写侧（Config 属性、自动下线改写、节点发现）必须共用同一份，
各写一份就会出现"写 A 读 B"（历史上先后踩过叠加层与 registry 两次）。
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from .model_registry import WINDOW_SIZE_SUFFIX  # "什么算窗口标注"的规则与清单侧共用一份

ROOT = Path(__file__).resolve().parent.parent
# ${VAR} 取环境变量；${VAR:-默认值} 缺失时用默认值（不告警）。
# 默认值语法是为了「同一份 config.yaml 既能开箱即用、又能被 .env 覆盖」：
# 例如 collaboration.cooldown_base: ${COLLAB_COOLDOWN_BASE:-10}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}")
# 启动时已打印过的缺失变量，避免同一变量在嵌套结构中重复警告
_missing_logged: set[str] = set()
# 已警告过的"类型写错"配置项，避免同一 key 每次读取都刷屏
_bad_value_logged: set[str] = set()
# 已点名过的"引用环境变量却解析为空"的站+字段（见 _warn_empty_ref）
_empty_ref_logged: set[str] = set()
# 免费节点叠加层默认路径（自动发现的增量写这里，绝不碰 git 跟踪的 model_registry.txt）
_DEFAULT_OVERLAY = "data/free_nodes.json"


def _warn_bad_value(key: str, value: Any, default: Any) -> None:
    """配置值类型异常时告警一次（走 stderr，保持 stdout 纯净约定）。"""
    if key in _bad_value_logged:
        return
    print(
        f"[配置警告] {key} 的值 {value!r} 不是合法数字，已按默认值 {default} 处理",
        file=sys.stderr,
    )
    _bad_value_logged.add(key)


def _warn_bad_bool(key: str, value: Any, default: Any) -> None:
    """布尔配置值异常时告警一次（与 _warn_bad_value 同源，提示语区分类型）。"""
    if key in _bad_value_logged:
        return
    print(
        f"[配置警告] {key} 的值 {value!r} 不是合法布尔（true/false/1/0），已按默认值 {default} 处理",
        file=sys.stderr,
    )
    _bad_value_logged.add(key)


def _interpolate(value: Any) -> Any:
    """递归把字符串里的 ${VAR} / ${VAR:-默认值} 替换为环境变量值。

    取值规则：
    - 环境变量存在（含空串）→ 用它的值；
    - 不存在且有 `:-默认值` → 用默认值，**不告警**（这是用户明确写了兜底）；
    - 不存在且无默认值 → 替换为空串并打印一次警告。

    设计选择：缺失 key 不直接报错，让启动能走完；但打印警告提醒用户配置可能不完整，
    避免 API Key 为空时静默走到运行时才炸。
    警告必须走 stderr：bridge 协议与 CLI --stream 的 stdout 是纯 JSON 行，
    任何混入 stdout 的提示都会让按行 json.loads 的调用方解析失败。
    """

    def _sub(m: re.Match) -> str:
        key, default = m.group(1), m.group(2)
        val = os.environ.get(key)
        if val is not None:
            return val
        if default is not None:
            return default
        if key not in _missing_logged:
            print(
                f"[配置警告] ${key} 未在 .env / model_registry.txt / 环境变量中定义，插值为空串",
                file=sys.stderr,
            )
            _missing_logged.add(key)
        return ""

    if isinstance(value, str):
        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def _split_models(raw: Any) -> list[tuple[str, int]]:
    """把「一站多模型」字段拆成 (模型名, 上下文长度) 列表，支持临时屏蔽。

    屏蔽机制（中转站免费模型常变，注释即下线）：
    - 列表中 # 前缀的项跳过，如 "a,#b,c" → ["a","c"]
    - YAML 里行首 # 注释的项 YAML 本身就会忽略
    上下文长度：用 "@长度" 标注单模型窗口，如 "gpt-4o-mini@128k,claude-3-haiku@200000"，
    不带 @ 的模型上下文长度记为 0（表示未指定，按默认处理）。
    长度单位缩写：k/K×1024，其余按整数 token 数。
    """
    if isinstance(raw, list):
        raw = ",".join(str(x) for x in raw)
    out: list[tuple[str, int]] = []
    for part in str(raw or "").split(","):
        p = part.strip()
        if not p or p.startswith("#"):  # 空项或 # 前缀 = 临时屏蔽
            continue
        # 取最后一个 @ 拆「窗口标注」；但模型名本身可能带 @（如 Cloudflare 的 @cf/...）。
        # 只有当 @ 之后是纯数字（可带 k/K）时才算窗口标注，否则 @ 属于模型名。
        # 判定规则由 model_registry.WINDOW_SIZE_SUFFIX 统一提供，与清单侧 _bare_model 同一条。
        name, sep, ctx = p.rpartition("@")
        ctx_len = 0
        if sep and WINDOW_SIZE_SUFFIX.fullmatch(ctx.strip()):  # @ 之后是上下文长度 → 拆窗口
            s = ctx.strip().lower()
            mul = 1024 if s.endswith("k") else 1
            num = s[:-1] if s.endswith("k") else s
            ctx_len = int(num) * mul  # 正则已保证是纯数字，不会抛
            name = name.strip()
        else:  # @ 是模型名一部分（@cf/...），或无 @ 标注：整串即模型名
            name = p
        if name:
            out.append((name, ctx_len))
    return out


def _warn_empty_ref(station: str, field: str, raw_value: Any, value: str) -> None:
    """`${VAR}` 引用解析成空串时点名"哪个站的哪个字段"（只报名字，绝不回显值）。

    为什么要在加载时就点名：变量名打错或 `.env` 里留成 `KEY=`（空值）时，插值只报一行
    "变量未定义"、Config 也不校验，症状要等到第一次派工才现形（整站 401/连不上）；而
    认证类失败是**终态不可重试**，会计入健康连击，攒够 `retire_days` 天就自动去给 git 跟踪的
    model_registry.txt 加 `#` —— 一个笔误被放大成清单改动。
    没引用环境变量的空串不点名：本地站（ollama）`api_key: ""` 是合法写法。
    """
    if not isinstance(raw_value, str) or "${" not in raw_value or value.strip():
        return
    key = f"{station}.{field}"
    if key in _empty_ref_logged:
        return
    _empty_ref_logged.add(key)
    print(
        f"[配置警告] 站 '{station}' 的 {field} 引用了环境变量，但解析结果为空："
        "检查 .env 里对应变量名是否拼对、是否写成了 `KEY=`（空值会顶掉 config.yaml 的 :-默认值）",
        file=sys.stderr,
    )


def _free_nodes_path(collab_cfg: dict) -> Path:
    """免费节点叠加层（data/free_nodes.json）路径的唯一口径。

    优先级：环境变量 `MAO_FREE_NODES_FILE`（测试隔离/临时指定）>
    `collaboration.free_nodes_file`（可对模型清单与密钥无感地改路径）> `_DEFAULT_OVERLAY`。
    写侧（`Config.free_nodes_path`，发现器落盘处）与读侧（`load_config` 套叠加层处）
    必须共用本函数：两处各写一份时曾出现"发现写到 A、启动读 B"，自动发现结果永不生效。
    值本身可能写成 `${COLLAB_FREE_NODES_FILE:-data/free_nodes.json}`，故先插值再判定绝对路径。
    """
    rel = os.environ.get("MAO_FREE_NODES_FILE") or str(
        _interpolate(collab_cfg.get("free_nodes_file", _DEFAULT_OVERLAY))
    )
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def _model_registry_path() -> Path:
    """模型清单文件（model_registry.txt）路径的唯一口径。

    优先级：环境变量 `MAO_MODEL_REGISTRY_FILE`（测试隔离 / 临时指定）> 项目根
    model_registry.txt。读侧（`load_config` 的基础层读取）与写侧（`Config.model_registry_path`，
    自动下线改写 / 节点发现）必须共用本函数：两处各写一份时，配了该覆盖的用户会"读 A 写 B"
    ——下线标记写进一个启动时根本不加载的文件，模型下线看似失效（与 `_free_nodes_path`
    修过的"发现写 A、启动读 B"同款教训）。
    """
    rel = os.environ.get("MAO_MODEL_REGISTRY_FILE") or "model_registry.txt"
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


# ---------- 数值配置的安全读取 ----------
# （2026-11-04 自 config.py 拆出，守「单文件 ≤500 行」约定；Config 以 staticmethod
#   形式 re-export，`Config.as_int(...)` / `self.as_int(...)` 的调用面逐字不变。）


def as_int(value: Any, key: str, default: int, minimum: int | None = None) -> int:
    """把配置值安全转成 int：非法值警告一次并回退默认值，不让程序崩。

    为什么不能裸 int()：配置写错类型（`max_participants: five`）会让程序在
    启动或首次访问该属性时直接抛 ValueError 崩掉，报错还是 "invalid literal
    for int()" 这种看不出是哪个配置项的信息。而本项目对配置缺 key 的一贯态度
    是"不阻启动、只警告"（见 _interpolate），类型错也该同样处理。
    """
    try:
        out = int(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError：int(float('inf')) 走的是这条（yaml 写 `cooldown_max: .inf`
        # 即触发，2026-11-04 审计实测崩启动）；nan 是 ValueError。
        _warn_bad_value(key, value, default)
        out = default
    return out if minimum is None else max(minimum, out)


def as_float(value: Any, key: str, default: float, minimum: float | None = None) -> float:
    """同 as_int，用于 float 型配置项。

    额外拒掉非有限值（inf/nan）：`latency_probe_interval: .inf` 这类写法
    float() 本身不报错，但会污染所有按它推导的限时/周期计算。
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        _warn_bad_value(key, value, default)
        out = default
    if not math.isfinite(out):
        _warn_bad_value(key, value, default)
        out = default
    return out if minimum is None else max(minimum, out)


def as_list(value: Any, key: str, default: list) -> list:
    """把配置值安全转成 list：非法值警告一次并回退默认值。

    与 as_int / as_float 同源：配置写错类型不该让程序崩，只警告并回退
    （本项目对配置的一贯态度是"不阻启动、只警告"）。
    单个字符串按单元素列表处理，方便 `workspace: ../proj` 这种简写。
    """
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value] if value.strip() else list(default)
    if isinstance(value, (list, tuple)):
        return list(value)
    _warn_bad_value(key, value, default)
    return list(default)


def as_bool(value: Any, key: str, default: bool) -> bool:
    """把配置值安全转成 bool：非法值警告一次并回退默认值，不让程序崩。

    接受 Python 布尔、0/1 数字，以及字符串 "true"/"false"/"yes"/"no"/"1"/"0"
    （不区分大小写）。其余值（如配置写成了 "省 token" 这类描述文本）视为
    类型错误：警告一次 + 回退默认，与 as_int / as_float 的态度一致。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
    _warn_bad_bool(key, value, default)
    return bool(default)
