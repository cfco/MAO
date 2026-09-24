"""配置加载：config.yaml 骨架 + ${ENV_VAR} 环境变量插值。

v4（2026-09-23 配置重构二期）：
    - config.yaml 只保留**结构骨架**（各段的值全部用 ${VAR} 引用，可带 :-默认值），
      不写任何真实运行值；
    - .env 承载**全部配置值**（端点 / KEY / LLM 参数 / 协作参数 / 工具参数），
      密级、不进 git；
    - model_registry.txt 承载**全部模型清单**（NODE_A_MODELS / LLM_MODEL 等），
      随仓库维护、git pull 即更新，也是自动下线机制唯一可改写区（给模型名加 #）。
    模型名与配置彻底分家：.env 里出现 *_MODEL(S) 会被分层守卫点名提醒。

配置分层（环境变量注入顺序，后层覆盖前层）：
    shell 环境变量 > .env（本地配置，含端点/key/参数）> model_registry.txt（模型清单基础层）
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
# 已警告过的"类型写错"配置项，避免同一 key 每次读取都刷屏
_bad_value_logged: set[str] = set()


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
            print(f"[配置警告] ${key} 未在 .env / model_registry.txt / 环境变量中定义，插值为空串", file=sys.stderr)
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
    # 能力标签（逗号分隔，如 "code,中文,长上下文"）：外部主据此决定把哪类活派给谁。
    tags: str = ""
    # 该模型清单来自 model_registry.txt 里的哪个变量（如 NODE_A_MODELS），
    # 便于诊断"这批模型挂在哪一站"，也是自动下线改写（加 #）的目标行定位用。
    # 值写死在 config.yaml（非 ${VAR}）时为空。
    models_env: str = ""

    def brief(self) -> dict:
        """对外展示用（不含 key）。"""
        return {
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "context_length": self.context_length or DEFAULT_CONTEXT,
            "note": self.note,
            "tags": self.tags,
        }


class Config:
    def __init__(self, data: dict, raw: dict | None = None):
        """data：已完成 ${VAR} 插值的配置；raw：插值前的原始数据（可选）。

        raw 用于反查「agents[*].models 来自哪个变量」（记录到 AgentProfile.models_env），
        模型健康自动下线时据此改写 model_registry.txt 的对应行。缺省（如测试直接构造）留空。
        """
        self.raw = raw if raw is not None else data  # 兜底：构造方没传 raw 时用 data，避免 None
        self.llm_cfg: dict = data.get("llm", {}) or {}
        self.tools_cfg: dict = data.get("tools", {}) or {}
        self.collab_cfg: dict = data.get("collaboration", {}) or {}
        self.mcp_servers: list = data.get("mcp_servers", []) or []

        # 站名 → 插值前的原始条目（只为提取 models_env，其余字段不依赖）
        # 读 self.raw 而非参数 raw：构造方未传 raw 时 self.raw 已兜底为 data，
        # 两处口径一致（审计 P2：曾有一个读 self.raw 一个读参数的半截兜底）。
        raw_items: dict[str, dict] = {}
        for item in (self.raw or {}).get("agents", []) or []:
            if isinstance(item, dict) and "name" in item:
                raw_items[str(item["name"])] = item

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
            # models 字段插值前形如 ${NODE_A_MODELS} → 提取变量名 NODE_A_MODELS
            raw_item = raw_items.get(station) or {}
            raw_models = str(raw_item.get("models") or raw_item.get("model") or "")
            m = _ENV_PATTERN.search(raw_models)
            models_env = m.group(1) if m else ""
            for mi, ctx_len in model_list:
                # 一站单模型保留站名（简洁），多模型加模型后缀保证可识别
                pname = station if len(model_list) == 1 else f"{station}:{mi}"
                if pname in seen:
                    continue
                seen.add(pname)
                self.agent_profiles.append(
                    AgentProfile(
                        name=pname,
                        base_url=base_url,
                        api_key=api_key,
                        model=mi,
                        context_length=ctx_len,
                        note=str(item.get("note", "")),
                        tags=str(item.get("tags", "")),
                        models_env=models_env,
                    )
                )

    # ---------- 智能体池 ----------

    def profile(self, name: str) -> AgentProfile | None:
        for p in self.agent_profiles:
            if p.name == name:
                return p
        return None

    # ---------- 数值配置的安全读取 ----------

    @staticmethod
    def as_int(value: Any, key: str, default: int, minimum: int | None = None) -> int:
        """把配置值安全转成 int：非法值警告一次并回退默认值，不让程序崩。

        为什么不能裸 int()：配置写错类型（`max_participants: five`）会让程序在
        启动或首次访问该属性时直接抛 ValueError 崩掉，报错还是 "invalid literal
        for int()" 这种看不出是哪个配置项的信息。而本项目对配置缺 key 的一贯态度
        是"不阻启动、只警告"（见 _interpolate），类型错也该同样处理。
        """
        try:
            out = int(value)
        except (TypeError, ValueError):
            _warn_bad_value(key, value, default)
            out = default
        return out if minimum is None else max(minimum, out)

    @staticmethod
    def as_float(value: Any, key: str, default: float, minimum: float | None = None) -> float:
        """同 as_int，用于 float 型配置项。"""
        try:
            out = float(value)
        except (TypeError, ValueError):
            _warn_bad_value(key, value, default)
            out = default
        return out if minimum is None else max(minimum, out)

    @staticmethod
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

    @staticmethod
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

    @property
    def max_workers(self) -> int:
        return self.as_int(self.collab_cfg.get("max_workers", 3), "collaboration.max_workers", 3,
                           minimum=1)

    @property
    def max_participants(self) -> int:
        """单次批量派工最多参与的工人数。<=0 表示不限（池内全部可用工人）。

        池子按「一站多模型」组织，很容易到几十个模型（model_registry.txt 里单站就挂了
        20 个）。"缺省用池内全部工人"在这种配置下等于一次打出几十个网络请求，
        而并发上限 max_workers 只有个位数：尾部批次必然撞上整体限时被判"未完成"
        丢弃——额度白烧、结果还拿不全。默认只取前 N 个（按池内确定性顺序、
        跳过冷却与当日隔离的工人），把派工规模与"免费额度 + 单轮时延"拉回可控范围。
        """
        return self.as_int(
            self.collab_cfg.get("max_participants", 5), "collaboration.max_participants", 5
        )

    @property
    def economy(self) -> bool:
        """省 token 开关（二元）：true=主智能体在派工协作时优先省自己的 token。

        语义边界：MAO 层只负责「启动时读配置作为初始值 + 允许外部主经
        bridge `set_economy` 指令运行期切换」，**绝不**在这里判断"当前任务
        该用分工还是重复验证"——模式判断全部留给外部主。此属性只在
        Bridge 构造时读取一次，作为运行态开关的初始值（见 bridge.py）。
        """
        return self.as_bool(
            self.collab_cfg.get("economy", False), "collaboration.economy", False
        )

    # ---------- 模型健康档案（collaboration.*） ----------

    @property
    def retire_days(self) -> int:
        """连续几个"运行日"出现**终态不可重试失败**后自动下线（在 model_registry.txt 加 #）。

        "运行日"= 程序实际启动并工作过的自然日（见 health.py active_dates）：
        没运行项目的那天不计入、也不打断连击。默认 7（collaboration.retire_days）。
        注意：下线只由"重试也没用"的终态失败（认证失败、非重试类 4xx）累积驱动；
        超时/429/5xx 等可重试瞬时失败只进短时冷却，永不计入此处、也就不会触发自动下线
        （见 health.py「由此对自动下线的影响」）。
        """
        return self.as_int(
            self.collab_cfg.get("retire_days", 7), "collaboration.retire_days", 7, minimum=1
        )

    @property
    def fallback_models(self) -> int:
        """单模型可重试失败后，最多尝试的额外模型数（0=关闭回退）。

        配合「一站多模型」：某模型节点抖动/限流时换同站其它模型往往能成，避免一次
        抖动废掉一整个免费额度。上限防止整站挂时把同站几十个模型挨个试（额度蔓延）。
        """
        return self.as_int(
            self.collab_cfg.get("fallback_models", 2), "collaboration.fallback_models", 2,
            minimum=0,
        )

    @property
    def fallback_cross_station(self) -> bool:
        """回退候选是否允许跨中转站（默认 True）：子智能体异常停止时从整池恢复。

        True：某工人可重试失败先用同站其它模型，同站无可用候选（或该站已被本批次
        探明故障）则再跨站、从整池挑「未试过且当前可用」的模型重试——即"异常停止了
        自动换模型恢复"。False：退回旧行为，只在同站内回退、绝不跨站（额度最省）。
        """
        return self.as_bool(
            self.collab_cfg.get("fallback_cross_station", True),
            "collaboration.fallback_cross_station", True,
        )

    @property
    def model_registry_path(self) -> Path:
        """模型清单文件（model_registry.txt）路径，自动下线改写（加 #）的目标文件。

        优先级：环境变量 MAO_MODEL_REGISTRY_FILE（测试隔离用）> 项目根 model_registry.txt。
        该文件承载全部模型名、随仓库维护，与 .env（配置）分开。
        """
        rel = os.environ.get("MAO_MODEL_REGISTRY_FILE") or "model_registry.txt"
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p

    @property
    def model_health_path(self) -> Path:
        """模型健康档案（当日终态失败隔离/连续运行日下线记录）落盘位置。

        优先级：环境变量 MAO_HEALTH_FILE > collaboration.health_file > 默认
        data/model_health.json。环境变量主要是给测试用：每个用例指到
        tmp_path，避免共享档案造成跨用例隔离污染。
        """
        rel = os.environ.get("MAO_HEALTH_FILE") or str(
            self.collab_cfg.get("health_file", "data/model_health.json")
        )
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p

    # ---------- 模型延迟档案（collaboration.*，问题5） ----------

    @property
    def model_latency_path(self) -> Path:
        """模型延迟档案（各工人最近往返耗时样本）落盘位置。

        优先级：环境变量 MAO_LATENCY_FILE（测试隔离）> collaboration.latency_file >
        默认 data/model_latency.json。只记延迟数值与更新时间，不含任何密钥。
        """
        rel = os.environ.get("MAO_LATENCY_FILE") or str(
            self.collab_cfg.get("latency_file", "data/model_latency.json")
        )
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p

    @property
    def latency_probe(self) -> bool:
        """是否启用延迟自动探测（默认 true）。

        为什么默认开：延迟档案全靠"每 latency_probe_interval 秒对全部节点（含当前
        不可用节点）探测一轮"刷新。不探测就永远没有延迟数据，pick() 的延迟选路等于
        空转（退回旧的"只按可用性选"）。
        环境变量 MAO_LATENCY_PROBE=0 可单独关闭（测试用：离线用例不允许后台线程打网络）。
        """
        env = os.environ.get("MAO_LATENCY_PROBE")
        if env is not None:
            return self.as_bool(env, "MAO_LATENCY_PROBE", True)
        return self.as_bool(
            self.collab_cfg.get("latency_probe", True), "collaboration.latency_probe", True
        )

    @property
    def latency_probe_interval(self) -> float:
        """延迟探测周期（秒，默认 600 = 10 分钟；下限 10s 防误配成高频探测烧额度）。"""
        return self.as_float(
            self.collab_cfg.get("latency_probe_interval", 600),
            "collaboration.latency_probe_interval", 600.0, minimum=10.0,
        )

    @property
    def latency_probe_timeout(self) -> float:
        """探测专用超时（秒，默认 8；下限 1s）——**不**沿用真实派工的 llm.timeout。

        为什么单列：探测只测"节点响应快不快"，一次极短往返就够。若沿用派工的
        llm.timeout（默认 120s）+ max_retries（默认 2），一个卡住/慢的坏节点会把一次
        "测速"拖成分钟级、并白打多次请求；更糟的是同步 probe（bridge/MCP `latency
        probe:true`）会被挂住整轮。故探测客户端用这个短超时且 max_retries=0（只发一次），
        失败只丢样本、不占派工额度。一轮最坏耗时 ≈ ceil(节点数/并发) × 本值。
        """
        return self.as_float(
            self.collab_cfg.get("latency_probe_timeout", 8),
            "collaboration.latency_probe_timeout", 8.0, minimum=1.0,
        )

    @property
    def latency_keep_ratio(self) -> float:
        """延迟裁剪后保留的比例（默认 0.6，夹在 0.1~1.0）。

        用户口径：可用模型 >5 个时"不使用高延迟的 40%，只用延迟较低的 60%"。
        同一条公式在小池上也自洽（n=5→3、4→3、3→2、2→2），恰好落在"可用低于 5 个
        就用延迟最低的 2~3 个"——故不另设小池分支，两档口径共用一套实现。
        """
        raw = self.as_float(
            self.collab_cfg.get("latency_keep_ratio", 0.6),
            "collaboration.latency_keep_ratio", 0.6, minimum=0.1,
        )
        return min(1.0, raw)

    @property
    def latency_min_samples(self) -> int:
        """至少有多少个候选工人的延迟已知，才启用延迟裁剪（默认 2）。

        冷启动保护：档案为空 / 刚部署时样本不足，绝不能凭空把工人踢出派工——证据不足
        就原样返回，等探测把样本补上再裁。
        """
        return self.as_int(
            self.collab_cfg.get("latency_min_samples", 2),
            "collaboration.latency_min_samples", 2, minimum=1,
        )

    # ---------- 兼容旧配置的兜底单模型 ----------

    @property
    def llm(self) -> dict:
        return self.llm_cfg

    # ---------- 常规 ----------

    @property
    def shell_timeout(self) -> int:
        return self.as_int(
            self.tools_cfg.get("shell_timeout", 120), "tools.shell_timeout", 120, minimum=1
        )

    @property
    def workspace(self) -> list[str]:
        """额外允许工具访问的工作区目录（tools.workspace）。

        为什么需要：MAO 的定位是「谁启动谁当主」，但它经常被用来驱动**别的项目**
        （在 MAO 里分析/改造另一个仓库）。工具层原先把所有路径硬绑在 MAO 自己的
        项目根上，主智能体连目标项目的一个文件都读不到 —— "驱动外部项目"就成了空话。

        安全默认：不配即空列表 = 仅限本项目根，原有安全边界完全不变。
        相对路径基于项目根解析，绝对路径原样使用。

        示例：
          tools:
            workspace: ["../other-proj", "D:/work/some-repo"]
        """
        raw = self.as_list(self.tools_cfg.get("workspace"), "tools.workspace", [])
        return [s for s in (str(x).strip() for x in raw) if s]


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
            # 剥行内注释：值里以"空格+#"开头的部分视为注释（如 `TOOLS_SHELL_TIMEOUT=120 # 秒`）。
            # 不剥模型清单的屏蔽标记：`NODE_A_MODELS=#m-one` 的 # 紧贴值无前导空格，不受影响。
            val = val.split(" #", 1)[0].strip().strip('"').strip("'")
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
    global _cfg_cache, _cfg_cache_mtime, _cfg_cache_dotenv_mtime, _cfg_cache_registry_mtime
    global _dotenv_keys
    p = path or (ROOT / "config.yaml")
    cacheable = path is None
    registry_path = ROOT / "model_registry.txt"
    dotenv_path = ROOT / ".env"
    _load_dotenv(registry_path)          # 基础层：模型清单
    env_names = _load_dotenv(dotenv_path)  # 配置层：端点/KEY/参数

    cfg_mtime = _safe_mtime(p)
    dotenv_mtime = _safe_mtime(dotenv_path)
    registry_mtime = _safe_mtime(registry_path)
    if (
        cacheable
        and not force
        and _cfg_cache is not None
        and _cfg_cache_mtime == cfg_mtime
        and _cfg_cache_dotenv_mtime == dotenv_mtime
        and _cfg_cache_registry_mtime == registry_mtime
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
    injected: set[str] = set()
    _load_dotenv(registry_path, injected)
    env_names = _load_dotenv(dotenv_path, injected)
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
    data: dict = {}
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cfg = Config(_interpolate(data), raw=data)
    if not cacheable:
        return cfg  # 显式路径：不进全局缓存，避免污染/错拿默认配置的缓存
    _cfg_cache = cfg
    _cfg_cache_mtime = cfg_mtime
    _cfg_cache_dotenv_mtime = dotenv_mtime
    _cfg_cache_registry_mtime = registry_mtime
    return _cfg_cache


# ---------- 进程内配置缓存（热加载用）----------
_cfg_cache: Config | None = None
_cfg_cache_mtime: float = 0.0             # config.yaml 的 mtime，变化即重载
_cfg_cache_dotenv_mtime: float = 0.0      # .env 的 mtime，变化即重载
_cfg_cache_registry_mtime: float = 0.0    # model_registry.txt 的 mtime，变化即重载
_dotenv_keys: set[str] = set()            # 由 dotenv 文件注入的变量名（热加载时需更新）
