"""配置对象：把插值后的 config.yaml 映射成可读属性（池 / 数值 / 路径 / 开关）。

本文件只保留 **Config 装配与读取面**。周边已按职责分家（2026-10-19，原单文件 806 行
超「单文件 ≤500 行」约定）：
  config_util.py —— 无状态工具：${VAR} 插值、模型清单拆分、路径口径、类型容错告警
  profiles.py    —— AgentProfile（池成员定义）+ DEFAULT_CONTEXT
  config_load.py —— 加载流程：分层 dotenv、叠加层套用、进程内缓存（load_config）

三者的公共名字仍从本模块 re-export，历史导入面（`from .config import Config,
load_config, ROOT, _interpolate`）一律不用改。

配置分层（环境变量注入顺序，后层覆盖前层）：
    shell 环境变量 > .env（本地配置，含端点/key/参数）> model_registry.txt（模型清单基础层）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# re-export：加载流程与工具函数历史上都在本文件内，测试/外壳按名取（F401 是有意的）
from .config_load import _load_dotenv, load_config  # noqa: F401
from .config_util import (  # noqa: F401
    _DEFAULT_OVERLAY,
    _ENV_PATTERN,
    ROOT,
    _bad_value_logged,
    _free_nodes_path,
    _interpolate,
    _missing_logged,
    _model_registry_path,
    _split_models,
    _warn_bad_bool,
    _warn_bad_value,
)
from .profiles import DEFAULT_CONTEXT, AgentProfile  # noqa: F401


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
            discovery = str(raw_item.get("discovery") or "").strip()
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
                        discovery=discovery,
                    )
                )

    # ---------- 智能体池 ----------

    def profile(self, name: str) -> AgentProfile | None:
        """按名字取池成员（线性查找）。

        生产里的热路径走 `WorkerPool.profiles`（dict，O(1)）；本方法是配置层的查询面，
        供测试断言与"只想读配置、不想建池"的程序化调用方使用——不要因为 grep 生产代码
        没有调用点就删掉它。
        """
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
        return self.as_int(
            self.collab_cfg.get("max_workers", 3), "collaboration.max_workers", 3, minimum=1
        )

    @property
    def max_per_station(self) -> int:
        """同一中转站（base_url + api_key）同时允许**在飞**的 LLM 请求数，默认 1。

        为什么默认 1（"一站同时只跑一个模型"）：一站多模型只是同一份配额、同一条网关上
        的多个入口，并发打过去并不会更快——请求在站端排队互相拖慢（实测同站 3 路并发时
        单模型耗时数倍膨胀），更糟的是**延迟档案**（`_record_latency` 记的就是负载下的真实
        往返）会被污染成"这个模型慢"，而实际是站被自己人堵了；选路裁剪正是按这份档案做的，
        等于自己给自己制造错误的排名依据。串行后并行度由**站数**决定：跨站才真并行。

        口径：本值是**每站**配额，与全局 `max_workers` 取交集（见
        `ParallelMixin._batch_concurrency`）——2 站 × 1 仍只有 2 路在飞，哪怕
        max_workers=3。设成较大值即退回旧的"只看总并发、不看站"行为。
        """
        return self.as_int(
            self.collab_cfg.get("max_per_station", 1), "collaboration.max_per_station", 1, minimum=1
        )

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
        return self.as_bool(self.collab_cfg.get("economy", False), "collaboration.economy", False)

    # ---------- 投票与评审 ----------

    @property
    def vote_preview_chars(self) -> int:
        """投票阶段每个候选方案给投票人看的预览长度（字符，默认 240，夹在 40~2000）。

        为什么可配：预览太短（曾固定 80）时，投票人只能凭开头几十字判断，长方案之间
        的差异被截掉，投票退化成"选标题"；太长则 N 份候选 × 长度直接顶爆投票 prompt
        （免费小模型上下文有限）。默认 240 是"能看出结构差异、又不至于压垮上下文"的折中。
        """
        raw = self.as_int(
            self.collab_cfg.get("vote_preview_chars", 240),
            "collaboration.vote_preview_chars",
            240,
            minimum=40,
        )
        return min(2000, raw)

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
            self.collab_cfg.get("fallback_models", 2),
            "collaboration.fallback_models",
            2,
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
            "collaboration.fallback_cross_station",
            True,
        )

    @property
    def model_registry_path(self) -> Path:
        """模型清单文件（model_registry.txt）路径，自动下线改写（加 #）的目标文件。

        优先级与读侧统一走 `_model_registry_path()`：环境变量 MAO_MODEL_REGISTRY_FILE
        （测试隔离 / 临时指定）> 项目根 model_registry.txt。该文件承载全部模型名、
        随仓库维护，与 .env（配置）分开。
        """
        return _model_registry_path()

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
            "collaboration.latency_probe_interval",
            600.0,
            minimum=10.0,
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
            "collaboration.latency_probe_timeout",
            8.0,
            minimum=1.0,
        )

    @property
    def latency_probe_max_tokens(self) -> int:
        """探测请求的输出上限（token，默认 8，<=0 = 不带该参数）。

        为什么带：不带 max_tokens 时，样本里混进了"这个模型爱写多长"——一个话痨模型
        在同等节点速度下会被系统性地记成"更慢"，而延迟档案正是选路裁剪的排名依据。
        免费网关对未知参数行为不一，故这里是**可关**的软约束：探测侧遇到"参数不支持"
        类 4xx 会自动按站降级为不带该参数（见 LatencyMixin._probe_one），不会整轮丢样本。
        """
        return self.as_int(
            self.collab_cfg.get("latency_probe_max_tokens", 8),
            "collaboration.latency_probe_max_tokens",
            8,
            minimum=0,
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
            "collaboration.latency_keep_ratio",
            0.6,
            minimum=0.1,
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
            "collaboration.latency_min_samples",
            2,
            minimum=1,
        )

    # ---------- 免费节点自动发现（collaboration.*） ----------

    @property
    def free_node_discovery(self) -> bool:
        """是否启用免费节点自动发现（默认 true）。false=完全关闭，registry 原样生效。"""
        return self.as_bool(
            self.collab_cfg.get("free_node_discovery", True),
            "collaboration.free_node_discovery",
            True,
        )

    @property
    def free_node_interval(self) -> float:
        """自动发现周期（秒，默认 86400=每天；<=0 关闭定时，仅手动 refresh_nodes 可触发）。"""
        return self.as_float(
            self.collab_cfg.get("free_node_interval", 86400),
            "collaboration.free_node_interval",
            86400.0,
            minimum=0.0,
        )

    @property
    def free_node_timeout(self) -> float:
        """单站发现请求的超时（秒，默认 10；下限 1s）——不能沿用派工 llm.timeout，
        否则一个卡住的站会把发现轮拖成分钟级。抓取失败只跳过该站、不动叠加层。"""
        return self.as_float(
            self.collab_cfg.get("free_node_timeout", 10),
            "collaboration.free_node_timeout",
            10.0,
            minimum=1.0,
        )

    @property
    def free_nodes_path(self) -> Path:
        """免费节点叠加层（data/free_nodes.json）路径：自动发现的增量只写这里，
        绝不改 git 跟踪的 model_registry.txt。优先级：环境变量 MAO_FREE_NODES_FILE >
        collaboration.free_nodes_file > 默认 data/free_nodes.json。"""
        return _free_nodes_path(self.collab_cfg)

    # ---------- 常规 ----------

    @property
    def shell_timeout(self) -> int:
        return self.as_int(
            self.tools_cfg.get("shell_timeout", 120), "tools.shell_timeout", 120, minimum=1
        )

    @property
    def mcp_tool_timeout(self) -> float:
        """单次 MCP 工具调用的最长等待（秒，默认 300，下限 5）。

        原先硬编码 300：长事务型工具（跑测试、等构建）会被直接判超时，而短工具又白等
        同一个上限。抽成配置项后按接入的 server 类型自行调；超时只影响这一次调用
        （session 作废、后台走重连），不会永久摘掉该 server。
        """
        return self.as_float(
            self.tools_cfg.get("mcp_tool_timeout", 300),
            "tools.mcp_tool_timeout",
            300.0,
            minimum=5.0,
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
