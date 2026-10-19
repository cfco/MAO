"""配置加载流程：分层 dotenv 注入 + 叠加层套用 + 进程内缓存。

从 config.py 拆出（2026-10-19，为守「单文件 ≤500 行」约定）：本模块只负责"怎么把
磁盘上的 config.yaml / .env / model_registry.txt / data/free_nodes.json 变成一个
Config 对象"，不定义 Config 自身（在 config.py）、也不放插值/路径等纯工具（在
config_util.py）。

依赖方向：config.py → 本模块（末尾 re-export load_config），本模块**不**在顶层 import
Config（否则成环），改为在 load_config 内延迟导入——那时 config.py 早已定义完 Config。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from .config_util import (
    ROOT,
    _bad_value_logged,
    _free_nodes_path,
    _interpolate,
    _missing_logged,
    _model_registry_path,
)

if TYPE_CHECKING:  # 仅类型标注；运行期用函数内延迟导入（避免与 config.py 成环）
    from .config import Config


def _load_dotenv(path: Path, into: set[str] | None = None) -> list[str]:
    """把一个 KEY=VALUE 文件注入 os.environ（供 ${VAR} 插值使用），返回文件里出现的变量名。

    只返回名字、不返回值：调用方据此做分层体检（如 .env 里混入 *_MODELS 模型变量），
    返回值设计上就拿不走任何密钥内容。

    `into`（可选）收集**本次实际注入**的 key：热重载时用于回收"上次由 dotenv
    注入、但本次文件已删除"的 key（见 load_config），保证从 .env 删掉一行后
    os.environ 不再残留旧值。只对等于 os.environ 时才收集；shell 预置的
    同名变量（`key in os.environ and key not in _dotenv_keys` 分支）不收集、
    load_config 也不会回收它。

    分层加载（调用方按序调两次，后层覆盖前层同名变量）：
    1) model_registry.txt —— 基础层：模型清单（随仓库维护，进 git，自动下线改写它）；
    2) .env         —— 配置层：端点 / KEY / LLM 与协作参数（保密，不进 git）。
    优先级：shell 启动前已 export 的变量 > .env > model_registry.txt。

    保密约定：.env 以点开头，AI 不读取其内容；这里只负责在运行时加载，
    不打印、不落盘、不返回任何值。缺失的文件不影响启动（静默跳过，返回空列表）。

    热加载语义：
    - 进程启动前已在 shell 里 export 的变量优先级最高（不被 dotenv 文件覆盖）；
    - 由本函数注入的变量（记录在 _dotenv_keys）在文件变化时会被更新，
      避免「只改 .env 不改 config.yaml」时旧值被 setdefault 钉死、缓存永远不刷新。
    """
    global _dotenv_keys
    names: list[str] = []
    if not path.exists():
        return names
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            # 引号优先于注释剥离（审计：旧实现先 split(" #") 再 strip('"')）：
            # `KEY="a #b"` 的 # 属于值本身，旧写法会把它静默截成 `a`。
            val = val.strip()
            if val[:1] in ('"', "'"):
                quote = val[0]
                end = val.find(quote, 1)
                # 闭合引号之后的内容（` " # 备注`）视为注释丢弃；未闭合则取起始引号之后
                val = val[1:end] if end != -1 else val[1:]
            else:
                # 未加引号时才剥行内注释：值里以"空格+#"开头的部分视为注释（如
                # `TOOLS_SHELL_TIMEOUT=120 # 秒`）。不剥模型清单的屏蔽标记：
                # `NODE_A_MODELS=#m-one` 的 # 紧贴值无前导空格，不受影响。
                val = val.split(" #", 1)[0].strip()
            if not key:
                continue
            names.append(key)
            # 区分"shell 预置"与"dotenv 注入"：前者不覆盖，后者每次都更新
            # （后者规则同时保证分层顺序：先读 model_registry.txt 后读 .env，.env 覆盖 registry）
            if key in os.environ and key not in _dotenv_keys:
                continue  # shell 里已有，尊重 shell 的值
            os.environ[key] = val
            _dotenv_keys.add(key)
            if into is not None:
                into.add(key)
    except OSError:
        pass  # 读取失败不拖垮启动，交给原环境变量
    return names


def _safe_mtime(p: Path) -> float:
    """取文件 mtime，不存在返回 0.0（不抛异常，方便缓存比较）。"""
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def load_config(path: Path | None = None, force: bool = False) -> Config:
    """加载 config.yaml 并做环境变量插值。

    设计：
    - 分层 dotenv：先 model_registry.txt（模型清单基础层）再 .env（配置层：
      端点/KEY/参数），同名变量后者胜出、shell 预置最高（见 _load_dotenv）。
    - 带进程内缓存，config.yaml / .env / model_registry.txt 三者 mtime 均未变则直接复用。
      （model_registry.txt 参与运行时配置——模型健康下线会改写它，必须纳入失效判断；
      .env 由用户/AI 改值时同样要让缓存失效。）
    - force=True 强制重读并重新构建 Config（热加载场景）。
    - 缓存只服务于默认路径：显式传 path（测试/多配置场景）每次真实读盘，
      既不读缓存也不写缓存（审计 P2：旧实现缓存全局唯一却不区分 path，
      加载过一次显式路径后，默认路径的调用可能拿到别的文件构建的缓存）。
    - 返回新 Config 对象，调用方用它替换自己的 cfg 引用；已有 Agent 持有的旧 cfg
      不受影响，避免 MCP 连接被意外重连。

    调用方可频繁调用本函数，只有配置真正变化时才解析。
    """
    # 延迟导入：config.py 末尾会 re-export 本函数，顶层 import 会形成环
    from .config import Config

    global \
        _cfg_cache, \
        _cfg_cache_mtime, \
        _cfg_cache_dotenv_mtime, \
        _cfg_cache_registry_mtime, \
        _cfg_cache_overlay_mtime
    global _dotenv_keys
    p = path or (ROOT / "config.yaml")
    cacheable = path is None
    registry_path = _model_registry_path()
    dotenv_path = ROOT / ".env"
    # 两层 dotenv 只解析**一次**，注入与后面的回收/分层体检共用同一结果。
    # 旧实现在这里先无参解析一遍（只为拿 os.environ 供缓存判定用），确认要重建后又带
    # injected 再解析一遍：命中缓存时白解析（每次调用都吃两份磁盘 IO），未命中时
    # 第一遍的注入副作用又被第二遍覆盖，逻辑绕且容易改一处漏一处。
    injected: set[str] = set()
    _load_dotenv(registry_path, injected)  # 基础层：模型清单
    env_names = _load_dotenv(dotenv_path, injected)  # 配置层：端点/KEY/参数

    # config.yaml 先读出来：叠加层路径要取 collaboration.free_nodes_file（下面插值也用同一份）。
    data: dict = {}
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    # 免费节点叠加层：在插值前把"自动发现"的增量（added/retired）叠加到对应
    # NODE_*_MODELS 变量上。registry 仍是唯一真源，叠加层只做增量修正；
    # 抓取失败/MAO_FREE_NODE_PROBE=0 时 apply_overlay 整段跳过，离线安全。
    # 路径必须走 _free_nodes_path（与 Config.free_nodes_path 同一函数）：旧写法在这里只认
    # MAO_FREE_NODES_FILE 和硬编码默认，而写侧认 collaboration.free_nodes_file —— 配了该值的
    # 用户"发现写到 A、启动读 B"，发现结果永不生效。路径也不能只在 try 里赋值：import 一旦
    # 失败，下面 `_safe_mtime(overlay_path)` 就是 UnboundLocalError，启动直接崩。
    overlay_path = _free_nodes_path(data.get("collaboration") or {})
    try:
        from .core.node_discovery import apply_overlay as _apply_free_node_overlay

        _apply_free_node_overlay(overlay_path)
    except Exception:  # noqa: BLE001 - 叠加层读写异常绝不影响启动
        pass

    cfg_mtime = _safe_mtime(p)
    dotenv_mtime = _safe_mtime(dotenv_path)
    registry_mtime = _safe_mtime(registry_path)
    overlay_mtime = _safe_mtime(overlay_path)
    if (
        cacheable
        and not force
        and _cfg_cache is not None
        and _cfg_cache_mtime == cfg_mtime
        and _cfg_cache_dotenv_mtime == dotenv_mtime
        and _cfg_cache_registry_mtime == registry_mtime
        and _cfg_cache_overlay_mtime == overlay_mtime
    ):
        return _cfg_cache
    # mtime 变化 / 强制重载 / 首次加载：读磁盘 → 构建 Config → 更新缓存
    _missing_logged.clear()  # 新 config 可能引用不同的 ${VAR}，重置警告
    _bad_value_logged.clear()  # 同上：新配置可能已修好类型错，允许重新告警
    # 热重载回收：把「上次由 dotenv 注入、本次文件已删除」的 key 从 os.environ 清掉。
    # 旧实现 _dotenv_keys 只增不清，从 .env 删掉一行 key 后 os.environ 仍残留旧值，
    # "停用某 key" 永远不会生效（配置/安全语义泄漏）。回收集合只含**本次实际注入**
    # 的 key：shell 预置（不在 _dotenv_keys）的变量不受影响；registry/.env 两层任一
    # 层还有该 key 也不会被回收。
    for stale in _dotenv_keys - injected:
        os.environ.pop(stale, None)
    _dotenv_keys = injected
    # 分层守卫：模型名**必须**写在 model_registry.txt（随仓库维护、自动下线改写它），
    # .env 是配置区。若 .env 里混进模型清单变量，会静默盖住 registry 的 git 更新和
    # 自动下线（# 下线看似失效），必须显式提醒——只报变量名，绝不回显任何值。
    strays = [n for n in env_names if n.endswith("_MODEL") or n.endswith("_MODELS")]
    if strays:
        print(
            f"[配置分层提醒] .env 中出现模型清单变量：{', '.join(strays)}。"
            "模型名一律写在 model_registry.txt（随仓库维护、自动下线也只改写它），"
            ".env 里的同名变量会盖住 registry 的 git 更新和自动下线，"
            "导致模型新增/下线看似失效——建议删掉这几行。",
            file=sys.stderr,
        )
    cfg = Config(_interpolate(data), raw=data)
    if not cacheable:
        return cfg  # 显式路径：不进全局缓存，避免污染/错拿默认配置的缓存
    _cfg_cache = cfg
    _cfg_cache_mtime = cfg_mtime
    _cfg_cache_dotenv_mtime = dotenv_mtime
    _cfg_cache_registry_mtime = registry_mtime
    _cfg_cache_overlay_mtime = overlay_mtime
    return _cfg_cache


# ---------- 进程内配置缓存（热加载用）----------
_cfg_cache: Config | None = None
_cfg_cache_mtime: float = 0.0  # config.yaml 的 mtime，变化即重载
_cfg_cache_dotenv_mtime: float = 0.0  # .env 的 mtime，变化即重载
_cfg_cache_registry_mtime: float = 0.0  # model_registry.txt 的 mtime，变化即重载
_cfg_cache_overlay_mtime: float = 0.0  # data/free_nodes.json 的 mtime，变化即重载
_dotenv_keys: set[str] = set()  # 由 dotenv 文件注入的变量名（热加载时需更新）
