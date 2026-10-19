"""多智能体编排：工人池（子智能体）与派工。

核心理念（三个臭皮匠顶个诸葛亮）：
- 主智能体＝外部驱动方（经 bridge / mcp / call 三外壳调用本项目），负责拆解、派工、
  汇总、把关——本项目不再自带内部 Agent Loop（已随"单一 bridge 内核"收敛移除）。
- 工人池：智能体池里的全部成员，纯文本执行器（不挂本地工具，兼容性最好、最安全）。
- 派工能力：ask_result（单个，结构化）/ ask_many（并行 + 结构化）/ vote（两步投票取共识）；
  pick() 跨调用 round-robin 分摊负载、可跳过冷却/隔离、支持按 tags 选路，并按**延迟**
  排序裁剪（优先派给延迟低的可用模型，见 _latency_window）。
- 历史演进：ask 结果 LRU 缓存、并发同 key 去重、swarm 的 ask_worker 工具等均已移除
  （外部主按需发话、命中率极低）；每次如实打网络。

文件结构（2026-09-22 拆分、2026-09-23 二次拆分并行段、
2026-10-19 三次拆分单工派工 / 批量 / 免费节点后发现，本文件回到 ≤500 行）：
  orchestrator.py —— WorkerPool 类装配：池状态 + pick 轮转 + 健康/冷却记账 + 客户端缓存
  dispatch.py     —— DispatchMixin：单工人派工（ask_result / ask / 回退 / 门外重试）
  batch.py        —— BatchMixin：批量派工（collect / _gather_many / ask_many / ask_many_structured）
  free_nodes.py   —— FreeNodeMixin：免费节点发现与热同步
  station_gate.py —— StationGateMixin：同站串行门（一站同时只跑一个模型）+ 并发口径
  parallelism.py  —— ParallelMixin：并行派工（_wait_gather / _effective_timeout / _run_parallel）
  voting.py       —— VotingMixin：投票两步（收集 + 编号投票）、择优
  constants.py    —— 派工结果状态码（ST_OK/ST_ERROR 等）
  health.py       —— 模型健康档案（当日失败隔离）
  latency.py      —— 模型延迟档案 + 定时探测器（选路用"快不快"）
  llm.py          —— LLM 适配层
"""

from __future__ import annotations

import sys
import threading
import time

from ..config import AgentProfile, Config
from .batch import BatchMixin
from .dispatch import DispatchMixin
from .free_nodes import FreeNodeMixin
from .health import ModelHealth, get_health
from .latency import LatencyMixin, LatencyStore
from .llm import LLMClient
from .node_discovery import FreeNodeDiscoverer
from .parallelism import ParallelMixin
from .station_gate import StationGateMixin
from .voting import VotingMixin


class WorkerPool(
    StationGateMixin,
    LatencyMixin,
    ParallelMixin,
    VotingMixin,
    DispatchMixin,
    BatchMixin,
    FreeNodeMixin,
):
    """子智能体池：纯文本执行器（不挂本地工具，兼容性最好、最安全）。

    免费节点友好：单节点抖动/限流/不可用不拖垮整体——连续失败进入**短时冷却**，
    冷却中跳过该节点（ask 直接返回冷却提示），其它节点照常工作，冷却到期后自动回来。
    健康档案（ModelHealth）叠加在冷却之上：仅当某模型出现**终态不可重试错误**
    （认证失败、非重试类 4xx）才当天隔离（当天不再派工）；可重试的瞬时失败只进冷却，
    不会升级成全天封禁——见 _record_result。

    公共 API 面（#4）：三外壳（bridge/mcp/call）实际只用到 ask_result /
    ask_many_structured / vote / collect / pick / health_status。此外 ask / ask_many
    （文本报告版，二者是结构化结果的渲染封装）、select_best（对**已有候选**投票择优，
    是 _run_ballot 的公共出口）、names 是有意保留的程序化公共 API——供直接以 Python
    方式复用内核的调用方与单测使用，不是待清理的死代码。

    本文件只放「池状态装配 + pick 轮转 + 健康/冷却记账 + 客户端缓存」；派工、批量、
    免费节点发现各自拆到独立 Mixin（见文件头文件结构），它们通过 self 共享本类的
    profiles / cfg / health 等属性，调用签名完全不变。
    """

    def __init__(
        self,
        cfg: Config,
        exclude: str | None = None,
        health: ModelHealth | None = None,
        latency: LatencyStore | None = None,
    ):
        self.cfg = cfg
        self.profiles: dict[str, AgentProfile] = {
            p.name: p for p in cfg.agent_profiles if p.name != exclude
        }
        # 模型健康档案：默认按配置落盘（data/model_health.json），测试可注入独立实例。
        # 派工失败时据此做当日隔离（当天不再派工给终态失败的模型）与连续运行日下线
        # （在 model_registry.txt 给模型加 #，见 health.py）。
        # 默认构造走 get_health()：同一档案路径全进程共享一个实例，避免多个工人池各持
        # 独立内存账本互相覆盖写盘、且彼此看不到隔离结果（见 health.get_health）。
        # registry_path（自动下线改写目标）与 retire_days（运行日阈值）一并传入。
        self.health = (
            health
            if health is not None
            else get_health(cfg.model_health_path, cfg.model_registry_path, cfg.retire_days)
        )
        # 模型延迟档案（问题5）：与 health 并列的一层——health 管"能不能用"，
        # 本档案管"快不快用"。同一批可用工人里优先派延迟低的（见 pick/_latency_window），
        # 并由 LatencyProber 每 latency_probe_interval 秒探测全部节点刷新它。
        # 具体构造在 LatencyMixin._init_latency（延迟选路整段外置到 latency.py）。
        self._init_latency(cfg, latency)
        # 下线改写需要模型名与清单变量名（如 NODE_A_MODELS），一次性建好映射
        self._model_of = {n: p.model for n, p in self.profiles.items()}
        self._env_of = {n: p.models_env for n, p in self.profiles.items()}
        # 节点健康：连续失败达阈值进入冷却，冷却期内跳过该节点。
        # 冷却时长指数上升，成功即清零。
        # name -> 冷却到期时刻。计时一律走 time.monotonic：冷却是"进程内还要等多久"的
        # 时限，用墙钟的话 NTP 校时/时区切换回拨一小时，就把一个 10s 冷却变成 1 小时
        # 空转（正向跳变则提前放行坏节点）。本表不进持久化，跨进程/重启无需可比性。
        self._cooldowns: dict[str, float] = {}
        self._fail_streak: dict[str, int] = {}  # name -> 连续失败次数
        # 本池已登记为"运行日"的那天（每天只登记一次，见 _note_run_day）
        self._active_day: str | None = None
        self._cooldown_threshold = self.cfg.as_int(
            self.cfg.collab_cfg.get("cooldown_fails", 2),
            "collaboration.cooldown_fails",
            2,
            minimum=1,
        )
        self._cooldown_base = self.cfg.as_float(
            self.cfg.collab_cfg.get("cooldown_base", 10),
            "collaboration.cooldown_base",
            10.0,
            minimum=0.0,
        )
        # 指数退避封顶（秒）：连续失败时长按 2 的幂翻倍，到顶后恒定，
        # 防止一个长期坏节点被算出天量冷却（否则 2^n 会很快溢出到数小时）。
        self._cooldown_max = self.cfg.as_int(
            self.cfg.collab_cfg.get("cooldown_max", 3600),
            "collaboration.cooldown_max",
            3600,
            minimum=1,
        )
        # 并发安全：ask_many/vote 用线程池并发调用 ask，健康状态与客户端缓存
        # （_clients）都需加锁保护。（历史上此处还有 LRU 答案缓存与并发在飞去重，
        # 已随「单一 bridge 路线」简化移除——外部主按需发话，命中率极低。）
        # 派工的单请求时长与重试次数：客户端构造与门外重试环共用同一份，别两处各解析各的
        self._llm_timeout = self.cfg.as_float(
            self.cfg.llm_cfg.get("timeout", 120), "llm.timeout", 120.0, minimum=1.0
        )
        self._max_retries = self.cfg.as_int(
            self.cfg.llm_cfg.get("max_retries", 2), "llm.max_retries", 2, minimum=0
        )
        self._lock = threading.Lock()
        # 按 profile.name 缓存 LLMClient，避免每次派工都重建连接池
        self._clients: dict[str, LLMClient] = {}
        # 探测专用客户端缓存：短超时 + 不重试，与真实派工的 _clients 分开（见 _probe_client_for）。
        # 绝不合用同一实例——派工要容忍节点慢/抖动（llm.timeout 120s + 重试），探测只求"最快摸
        # 一下在不在、快不快"，坏节点绝不能把测速拖成分钟级或打出多次请求，更会挂住同步 probe 整轮。
        self._probe_clients: dict[str, LLMClient] = {}
        # 客户端缓存独立一把锁：建 LLMClient 要建 httpx 客户端与 SSL 上下文（非纯内存操作），
        # 绝不能压在 self._lock 里——那会让并发首派把 _record_result / health_status 的
        # 纯内存记账一起串行化。两把锁互不嵌套（本缓存路径不碰记账，记账路径不建客户端）。
        self._clients_lock = threading.Lock()
        # pick() 轮转游标（#6）：随默认批量派工的调用推进，把负载摊到整个池，
        # 而非每次死取头部 N 个。单进程一池共享 → 全局 round-robin。
        # _pick_lock 单独一把：MCP 外壳下宿主可并发发工具调用（同步工具跑在线程池），
        # 多个默认派工会同时推进游标。绝不能复用 self._lock——pick()→_skip_reason()
        # →_is_in_cooldown() 已持 self._lock，重入非重入锁会自死锁。
        self._pick_cursor = 0
        self._pick_lock = threading.Lock()

        # 同站模型分组：按 (base_url, api_key) 把"一站多模型"聚到一起，供失败回退
        # 挑选同站其它可用模型。键用 (base_url, api_key) 而非仅 base_url：不同站可能
        # 共用同一 endpoint 但不同 key，按 key 区分更精确（同 key 才意味着"整站级故障共担"）。
        self._station_models: dict[tuple[str, str], list[str]] = {}
        for nm, p in self.profiles.items():
            self._station_models.setdefault(p.station_key, []).append(nm)

        # 同站串行门（collaboration.max_per_station，默认 1）：门表与闸门方法在
        # StationGateMixin（见 station_gate.py），卡在被真正发请求的那一层，
        # 批量派工 / 顺序 ask / 失败回退 / 延迟探测共用同一口径。
        self._init_station_gates(cfg)

        # 回退上限：单模型可重试失败后，最多尝试的额外模型数（0=关闭回退）。
        # 既利用"一站多模型"冗余，又限制额度蔓延（整站挂时不会把同站几十个模型挨个试）。
        self._fallback_models = self.cfg.fallback_models
        # 回退是否允许跨站（默认 true）：同站无可用候选时从整池恢复——"异常停止自动换模型"。
        self._fallback_cross_station = self.cfg.fallback_cross_station

        # 免费节点自动发现（叠加层 data/free_nodes.json）：建池时 best-effort 探一次
        # （守护线程、不阻塞建池、失败静默跳过），并按 free_node_interval 后台周期刷新
        # （懒式调度：挂在派工/预检入口顺带检查是否到期）。MAO_FREE_NODE_PROBE=0 或
        # free_node_discovery=false 或 interval<=0 → 不调度，registry 原样生效。
        self._free_node = FreeNodeDiscoverer(cfg, cfg.free_nodes_path)
        try:
            self._free_node.maybe_start()
        except Exception:  # noqa: BLE001 - 发现调度绝不影响建池
            pass

    # ---------- 基础派工 ----------

    def names(self) -> list[str]:
        return list(self.profiles)

    def pick(self, limit: int | None = None, tags: str | None = None) -> list[str]:
        """挑出本次批量派工要用的工人：跳过冷却与当日隔离，跨调用轮转分摊负载。

        缺省上限取 collaboration.max_participants（<=0 表示不限，返回全部可用）。

        为什么轮转（#6）：池子「一站多模型」轻易几十个，而固定取头部 N 个会让同一批
        头部节点反复挨打、最快被限流/隔离，尾部几十个模型整天闲置——既没摊薄免费额度
        又放大单点抖动。用一个随调用推进的游标做 round-robin：
        - 单次调用内仍是**连续窗口**（确定性）——投票编号、候选顺序不会在一次
          任务中途漂移；
        - 连续多次默认调用（不显式点名 workers）则轮流覆盖整个池，把负载摊到所有可用模型。
        首次调用从头部开始（游标 0），因此"池内前 N 个"仍是第一次的结果，向后依次轮转。

        延迟选路（问题5）：**缺省路径**（limit=None）先按延迟裁剪候选，再在裁剪后的
        子集里轮转——只用延迟最低的 60%（弃掉最高延迟的 40%；小池上同一公式落在
        "最低的 2~3 个"，见 _latency_window），并把候选按延迟升序排列，让窗口优先落
        在快节点上。显式传 limit（含 0=不限）表示调用方自己定规模，不做延迟裁剪。

        可选 `tags`（逗号分隔）：只保留能力标签**全部命中**的工人，供外部主按任务类型选路
        （标签见 list_agents / health 快照）。缺省 None 不过滤。
        """
        self._touch_latency()  # 到期就起一轮后台探测刷新延迟档案，不阻塞本次调用
        self._touch_free_nodes()  # 免费节点发现：到期就起一轮后台发现，不阻塞本次调用
        cap = self.cfg.max_participants if limit is None else limit
        live = [w for w in self.profiles if self._skip_reason(w) is None]
        if tags:
            need = {t.strip() for t in str(tags).split(",") if t.strip()}
            if need:

                def _has_tags(name: str) -> bool:
                    have = {
                        t.strip()
                        for t in str(getattr(self.profiles[name], "tags", "") or "").split(",")
                        if t.strip()
                    }
                    return need <= have

                live = [w for w in live if _has_tags(w)]
        if limit is None:
            live = self._latency_window(live)
        if cap is None or cap <= 0 or not live:
            return live
        n = len(live)
        with self._pick_lock:  # 仅护游标自增；live 快照与窗口切片在锁外，绝不嵌套 self._lock
            start = self._pick_cursor % n
            self._pick_cursor += min(cap, n)
        window = live[start:] + live[:start]  # 轮转后的连续窗口，仍保持池内相对顺序
        return window[:cap]

    def health_status(self) -> list[dict]:
        """每个工人的当前可用性快照，供外部主派工前预检（配合能力标签选路）。

        available=False 表示当前不可派：冷却中（cooldown_s 剩余秒）或当日失败隔离。
        这里只是给主的"该不该派"参考。
        latency_ms 为该工人的当前延迟（最近往返耗时中位数，毫秒）；None=尚无样本。
        字段语义上，available 是"能不能用"，latency_ms 是"快不快用"——两者独立：
        一个可用但很慢的节点仍然 available=True，只是默认派工会把它排在后面/裁掉。
        """
        self._touch_free_nodes()  # 派工/预检入口顺带触发免费节点发现（懒式，不阻塞）
        now = time.monotonic()
        lat = self.latency.snapshot()
        # 冷却表一次取全：逐工人取锁拼出来的"快照"，同一轮里各条目的时点会不一致
        # （先取的条目可能已被并发派工改过），与已修的"快照移入锁内取"同一族问题。
        with self._lock:
            untils = dict(self._cooldowns)
        out: list[dict] = []
        for name, p in self.profiles.items():
            until = untils.get(name, 0.0)
            quarantined = self.health.quarantined(name)
            out.append(
                {
                    "worker": name,
                    "model": p.model,
                    "tags": getattr(p, "tags", ""),
                    "available": until <= now and not quarantined,
                    "cooldown_s": round(max(0.0, until - now), 1),
                    "quarantined_today": quarantined,
                    "latency_ms": lat.get(name),
                }
            )
        return out

    # ---------- 冷却 / 当日隔离记账 ----------

    def _is_in_cooldown(self, name: str) -> bool:
        """该节点是否处于冷却期（冷却期跳过，避免反复打一个坏节点）。"""
        with self._lock:
            until = self._cooldowns.get(name, 0.0)
        return time.monotonic() < until

    def _skip_reason(self, name: str) -> str | None:
        """该工人当前不可派工的原因（'冷却中' / '当日失败隔离'），None=可派。

        缓存命中不受此限制（ask 里先查缓存），这里服务 ask_many/collect 的预过滤。
        """
        if self._is_in_cooldown(name):
            return "冷却中"
        if self.health.quarantined(name):
            return "当日失败隔离"
        return None

    def _record_result(self, name: str, ok: bool, fatal: bool = False) -> None:
        """更新节点健康状态：失败累计，达阈值进冷却（时长指数上升）；成功清零。

        冷却 vs 当日隔离分两层，按失败性质区分（#1/#2）：
        - 任何失败都累计连击、必要时进**短时冷却**（base×2^n 指数退避、封顶 cooldown_max，
          成功即清零、会自动恢复）；
        - 只有 **fatal=True（不可重试的终态错误：认证失败、非重试类 4xx 等）** 才记入
          持久健康档案（ModelHealth）→ **当天不再向该模型派工**。

        为什么这么分：免费中转站天天 timeout/429/5xx/连接抖动是常态，若"任何一次失败即
        全天隔离"，早上被限一次流就把整池打空、当天再不回来。可重试类失败交给冷却做
        短时退避即可，绝不该升级成全天封禁；只有"重试也没用"的终态错误才值得整天拉黑。

        档案落盘在池锁外做，失败路径不在热路径上，多一次小文件写无碍。
        """
        now = time.monotonic()
        succeeded = False
        pair = ("", "")
        with self._lock:
            if ok:
                self._cooldowns.pop(name, None)
                self._fail_streak[name] = 0
                succeeded = True
            else:
                streak = self._fail_streak.get(name, 0) + 1
                self._fail_streak[name] = streak
                if streak >= self._cooldown_threshold:
                    # 连续失败越多，冷却越久：指数退避 base × 2^(超出阈值的次数)，封顶 cooldown_max。
                    # 例（base=10, threshold=2）：第2次失败 10s、第3次 20s、第4次 40s……到顶后恒定。
                    # exponent 先封到 20：极端连击下避免计算 2^n 大整数（2^20×base 已远超合理封顶）。
                    exponent = min(streak - self._cooldown_threshold, 20)
                    duration = min(self._cooldown_base * (2**exponent), self._cooldown_max)
                    self._cooldowns[name] = now + duration
                pair = (self._model_of.get(name, ""), self._env_of.get(name, ""))
        if succeeded:
            self._note_run_day()  # 登记运行日同样是落盘 IO，放在池锁外
            return
        if not fatal:
            return  # 可重试的瞬时失败：只进冷却，不做当日隔离
        try:
            # 模型名与清单变量名要在池锁内取：`sync_free_nodes` 会整本换掉这两张表，
            # 锁外读可能拿到"记账时的名字 + 改写后的映射"这种错配，把 # 标到别的模型上。
            notice = self.health.record_failure(name, *pair)
        except Exception:  # noqa: BLE001 - 健康档案异常绝不影响派工主流程
            notice = None
        if notice:
            print(notice, file=sys.stderr)  # stderr：bridge/--stream 的 stdout 必须纯净

    def _note_run_day(self) -> None:
        """把今天登记为"运行日"（进程内每天一次，幂等）。

        为什么必须由成功派工来登记：health 的下线判定是"最近 retire_days 个**运行日**里
        每天都失败过"，而原先只有 `record_failure` 会往 active_dates 加日子 —— 模型**成功**
        的日子不进窗口，也就打断不了连击：相隔两个月各失败一天，攒够 3 天就被标 `#` 下线
        （实测复现）。登记制补上"这天程序确实跑过、而且这个模型干成了活"这一侧的事实。
        闸门放在内存里，避免每次成功派工都多写一次盘；登记失败不置闸门，下次成功派工重试。
        """
        today = time.strftime("%Y-%m-%d")
        if self._active_day == today:
            return
        try:
            self.health.note_active(today)
        except Exception:  # noqa: BLE001 - 记账失败绝不影响派工主流程
            return
        self._active_day = today

    # ---------- 客户端缓存 ----------

    def _client_for(self, p: AgentProfile) -> LLMClient:
        """按 profile.name 复用 LLMClient；首次访问时按工人配置创建并缓存。
        timeout / max_retries 从全局配置读取，保持与主智能体一致的重试策略。

        加锁创建：并发派工（ask_many/vote）下同一工人会被多个线程同时首次访问，
        不加锁会各建一个 LLMClient（多套连接池，其中一个被覆盖后无人引用）。
        缓存单走**独立的 _clients_lock**，不复用池记账锁 self._lock——LLMClient 构造要建
        httpx 客户端与 SSL 上下文（毫秒~几十毫秒级），压在 self._lock 里会让并发首派把
        _record_result / health_status 这些纯内存记账一起串行化。
        调用方（_lead）在调本方法时并未持有 self._lock，故不会自锁。
        """
        with self._clients_lock:
            client = self._clients.get(p.name)
            if client is not None:
                return client
            client = LLMClient(
                p.base_url,
                p.api_key,
                p.model,
                timeout=self._llm_timeout,
                max_retries=self._max_retries,
            )
            self._clients[p.name] = client
            return client

    def _probe_client_for(self, p: AgentProfile) -> LLMClient:
        """延迟探测专用客户端：短超时（cfg.latency_probe_timeout）且**不重试**（max_retries=0）。

        与 _client_for 分开缓存、绝不复用同一实例：真实派工要容忍节点慢/抖动（120s + 重试），
        探测只求"最快摸一下在不在、快不快"。若让探测继承派工超时策略，一个卡住的节点会把一次
        测速拖成分钟级、白打多次请求，更会挂住同步 probe（bridge/MCP `latency probe:true`）整轮。
        同样走 _clients_lock（与建/取真实派工客户端一把锁），chat 在锁外发起，不会与派工自锁。
        """
        with self._clients_lock:
            client = self._probe_clients.get(p.name)
            if client is None:
                client = LLMClient(
                    p.base_url,
                    p.api_key,
                    p.model,
                    timeout=self._latency_probe_timeout,
                    max_retries=0,
                )
                self._probe_clients[p.name] = client
            return client

    def _retire_worker_clients(
        self, removed: list[str], live_stations: set[tuple[str, str]]
    ) -> None:
        """回收被移出池的工人的连接池与站门（`sync_free_nodes` 的收尾，锁外调用）。

        为什么不能直接 close：被移出的工人可能**正在飞**（批量派工的 daemon 线程还会
        回调它），直接断开会把在飞请求打成异常；但也不能不回收——每轮 `refresh_nodes`
        都可能新增一批 LLMClient（各带一个 httpx 连接池），长期运行会持续累积。
        故走 `close_when_idle`：标记后由最后一个在飞请求收尾时关闭；空闲的当场就关。

        站门同理：只保留"池里还有模型的站"。已空的站门没人再取，摘掉即可
        （等待中的线程仍持有旧 Semaphore 引用，会正常跑完，不会被掐断）。
        """
        retired: list[LLMClient] = []
        # _clients_lock 只护缓存、不嵌套池锁：调用方已在池锁外（见 sync_free_nodes）
        with self._clients_lock:
            for name in removed:
                for cache in (self._clients, self._probe_clients):
                    client = cache.pop(name, None)
                    if client is not None:
                        retired.append(client)
        for client in retired:
            try:
                client.close_when_idle()
            except Exception:  # noqa: BLE001 - 回收失败不影响热同步结果
                pass
        with self._gates_lock:
            for station in [s for s in self._station_gates if s not in live_stations]:
                self._station_gates.pop(station, None)

    def close(self) -> None:
        """释放全部工人客户端连接池（含探测专用池，幂等）。池被弃用时由持有方调用。"""
        with self._clients_lock:
            clients = list(self._clients.values()) + list(self._probe_clients.values())
            self._clients.clear()
            self._probe_clients.clear()
        for c in clients:
            try:
                c.close()
            except Exception:  # noqa: BLE001 - 释放失败不应影响上层收尾
                pass
