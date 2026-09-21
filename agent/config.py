"""配置加载：config.yaml + ${ENV_VAR} 环境变量插值。

v2：智能体池（agents 列表）——每个条目是一个 AgentProfile。
主智能体=被启动方式选中/指定的那一条，其余自动成为工人（子智能体）。
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
# ${VAR} 取环境变量；${VAR:-默认值} 缺失时用默认值（不告警）。
# 默认值语法是为了「同一份 config.yaml 既能开箱即用、又能被 .env 覆盖」：
# 例如 llm.base_url: ${LLM_ENDPOINT:-https://api.deepseek.com/v1}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}")
# 启动时已打印过的缺失变量，避免同一变量在嵌套结构中重复警告
_missing_logged: set[str] = set()


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
            print(f"[配置警告] ${key} 未在 .env 或环境变量中找到，插值为空串", file=sys.stderr)
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
        name, _, ctx = p.partition("@")
        name = name.strip()
        ctx_len = 0
        if ctx:
            s = ctx.strip().lower()
            mul = 1024 if s.endswith("k") else 1
            if s.endswith("k"):
                s = s[:-1]
            try:
                ctx_len = int(float(s) * mul)
            except ValueError:
                ctx_len = 0
        if name:
            out.append((name, ctx_len))
    return out


DEFAULT_CONTEXT = 128_000  # 未标注上下文长度的模型按这个大小时钟估算裁剪历史
CONTEXT_SAFETY = 0.75       # 历史 token 上限取 上下文长度×该系数，给回复/工具调用留余量


@dataclass
class AgentProfile:
    """智能体池中的一条：一个可 API 接入的模型智能体。

    支持「一站多模型」：一个站（base_url+api_key）可挂多个 model，
    每个模型展开成一个 AgentProfile。名字：单模型取站名，多模型取"站名:模型名"。
    """

    name: str
    base_url: str
    api_key: str
    model: str
    context_length: int = 0  # 上下文窗口(token)；0=未指定按 DEFAULT_CONTEXT
    note: str = ""

    def brief(self) -> dict:
        """对外展示用（不含 key）。"""
        return {
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "context_length": self.context_length or DEFAULT_CONTEXT,
            "note": self.note,
        }


class Config:
    def __init__(self, data: dict):
        self.raw = data
        self.llm_cfg: dict = data.get("llm", {}) or {}
        self.server: dict = data.get("server", {}) or {}
        self.tools_cfg: dict = data.get("tools", {}) or {}
        self.collab_cfg: dict = data.get("collaboration", {}) or {}
        self.session_cfg: dict = data.get("session", {}) or {}
        self.mcp_servers: list = data.get("mcp_servers", []) or []

        self.agent_profiles: list[AgentProfile] = []
        seen: set[str] = set()
        for item in data.get("agents", []) or []:
            if not isinstance(item, dict) or "name" not in item:
                continue
            station = str(item["name"])  # 站名（唯一）
            base_url = str(item.get("base_url", ""))
            api_key = str(item.get("api_key", ""))
            # 兼容新结构 models（一站多模型）与旧结构 model（单模型）
            models = item.get("models") or item.get("model")
            model_list = _split_models(models)
            if not model_list:
                continue
            for m, ctx_len in model_list:
                # 一站单模型保留站名（简洁），多模型加模型后缀保证可识别
                pname = station if len(model_list) == 1 else f"{station}:{m}"
                if pname in seen:
                    continue
                seen.add(pname)
                self.agent_profiles.append(
                    AgentProfile(
                        name=pname,
                        base_url=base_url,
                        api_key=api_key,
                        model=m,
                        context_length=ctx_len,
                        note=str(item.get("note", "")),
                    )
                )

    # ---------- 智能体池 ----------

    def profile(self, name: str) -> AgentProfile | None:
        for p in self.agent_profiles:
            if p.name == name:
                return p
        return None

    @property
    def max_workers(self) -> int:
        return max(1, int(self.collab_cfg.get("max_workers", 3)))

    # ---------- 兼容旧配置的兜底单模型 ----------

    @property
    def llm(self) -> dict:
        return self.llm_cfg

    # ---------- 常规 ----------

    @property
    def shell_timeout(self) -> int:
        return int(self.tools_cfg.get("shell_timeout", 120))

    # ---------- 会话落盘限流（session.*） ----------

    @property
    def session_flush_batch(self) -> int:
        """待写缓冲累计达到该条数立即落盘（高频场景）。最小 1。

        优先级：环境变量 MAO_SESSION_FLUSH_BATCH > config.yaml 的 session.flush_batch > 默认 16。
        环境变量兜底用于 CLI 命令行覆盖（uv run mao chat --session-flush-batch N），
        同一进程内对 chat/run/pipeline/web/bridge 所有入口生效。
        """
        env = os.environ.get("MAO_SESSION_FLUSH_BATCH")
        if env is not None:
            try:
                return max(1, int(env))
            except ValueError:
                pass
        return max(1, int(self.session_cfg.get("flush_batch", 16)))

    @property
    def session_flush_interval(self) -> float:
        """距上次落盘超过该秒数才落盘（低频/跟随者场景）。最小 0.0。

        优先级：环境变量 MAO_SESSION_FLUSH_INTERVAL > config.yaml 的 session.flush_interval > 默认 2.0。
        """
        env = os.environ.get("MAO_SESSION_FLUSH_INTERVAL")
        if env is not None:
            try:
                return max(0.0, float(env))
            except ValueError:
                pass
        return max(0.0, float(self.session_cfg.get("flush_interval", 2.0)))

    @property
    def max_iterations(self) -> int:
        return int(self.llm_cfg.get("max_iterations", 25))

    @property
    def host(self) -> str:
        return str(self.server.get("host", "127.0.0.1"))

    @property
    def port(self) -> int:
        return int(self.server.get("port", 8000))


def _load_dotenv(path: Path) -> None:
    """把 .env 文件里的 KEY=VALUE 注入 os.environ（供 ${VAR} 插值使用）。

    保密约定：.env 以点开头，AI 不读取其内容；这里只负责在运行时加载，
    不打印、不落盘、不返回任何值。缺失的 .env 不影响启动（静默跳过）。

    热加载语义：
    - 进程启动前已在 shell 里 export 的变量优先级最高（不被 .env 覆盖）；
    - 由本函数从 .env 注入的变量（记录在 _dotenv_keys）在 .env 变化时会被更新，
      避免「只改 .env 不改 config.yaml」时旧值被 setdefault 钉死、缓存永远不刷新。
    """
    global _dotenv_keys
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if not key:
                continue
            # 区分"shell 预置"与".env 注入"：前者不覆盖，后者每次都更新
            if key in os.environ and key not in _dotenv_keys:
                continue  # shell 里已有，尊重 shell 的值
            os.environ[key] = val
            _dotenv_keys.add(key)
    except OSError:
        pass  # 读取失败不拖垮启动，交给原环境变量


def _safe_mtime(p: Path) -> float:
    """取文件 mtime，不存在返回 0.0（不抛异常，方便缓存比较）。"""
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def load_config(path: Path | None = None, force: bool = False) -> Config:
    """加载 config.yaml 并做环境变量插值。

    设计：
    - 带进程内缓存，默认 config.yaml 与 .env 的 mtime 均未变则直接复用。
      （只盯 config.yaml 不够：用户换 API key 通常只改 .env，旧插值会被缓存钉死。）
    - force=True 强制重读并重新构建 Config（热加载场景）。
    - 返回新 Config 对象，调用方用它替换自己的 cfg 引用；已有 Agent 持有的旧 cfg
      不受影响，避免 MCP 连接被意外重连。

    调用方（web/server._get_cfg）可频繁调用本函数，只有配置真正变化时才解析。
    """
    global _cfg_cache, _cfg_cache_mtime, _cfg_cache_dotenv_mtime
    p = path or (ROOT / "config.yaml")
    dotenv_path = ROOT / ".env"
    _load_dotenv(dotenv_path)

    cfg_mtime = _safe_mtime(p)
    dotenv_mtime = _safe_mtime(dotenv_path)
    if (
        not force
        and _cfg_cache is not None
        and _cfg_cache_mtime == cfg_mtime
        and _cfg_cache_dotenv_mtime == dotenv_mtime
    ):
        return _cfg_cache
    # mtime 变化 / 强制重载 / 首次加载：读磁盘 → 构建 Config → 更新缓存
    _missing_logged.clear()  # 新 config 可能引用不同的 ${VAR}，重置警告
    data: dict = {}
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    _cfg_cache = Config(_interpolate(data))
    _cfg_cache_mtime = cfg_mtime
    _cfg_cache_dotenv_mtime = dotenv_mtime
    return _cfg_cache


# ---------- 进程内配置缓存（热加载用）----------
_cfg_cache: Config | None = None
_cfg_cache_mtime: float = 0.0       # config.yaml 的 mtime，变化即重载
_cfg_cache_dotenv_mtime: float = 0.0  # .env 的 mtime，变化即重载
_dotenv_keys: set[str] = set()      # 由 .env 注入的变量名（热加载时需更新）
