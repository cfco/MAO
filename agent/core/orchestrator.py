"""多智能体编排：工人池（子智能体）与派工。

核心理念（三个臭皮匠顶个诸葛亮）：
- 主智能体＝外部驱动方（经 bridge / mcp / call 三外壳调用本项目），负责拆解、派工、
  汇总、把关——本项目不再自带内部 Agent Loop（已随"单一 bridge 内核"收敛移除）。
- 工人池：智能体池里的全部成员，纯文本执行器（不挂本地工具，兼容性最好、最安全）。
- 派工能力：ask_result（单个，结构化）/ ask_many（并行 + 结构化）/ vote（两步投票取共识）；
  pick() 跨调用 round-robin 分摊负载、可跳过冷却/隔离、支持按 tags 选路。
- 历史演进：ask 结果 LRU 缓存、并发同 key 去重、swarm 的 ask_worker 工具等均已移除
  （外部主按需发话、命中率极低）；每次如实打网络。

文件结构（2026-09-22 拆分后）：
  orchestrator.py —— WorkerPool 类：派工 + 并行 + 健康 + pick 轮转
  voting.py       —— VotingMixin：投票两步（收集 + 编号投票）、择优
  constants.py    —— 派工结果状态码（ST_OK/ST_ERROR 等）
  health.py       —— 模型健康档案（当日失败隔离）
  llm.py          —— LLM 适配层
"""
from __future__ import annotations

import concurrent.futures
import sys
import threading
import time
from concurrent.futures import Future, wait

from ..config import AgentProfile, Config
from .constants import (
    ST_COOLDOWN,
    ST_ERROR,
    ST_MISSING,
    ST_OK,
    ST_QUARANTINED,
    ST_TIMEOUT,
    WORKER_SYSTEM,
)
from .health import ModelHealth, get_health
from .llm import LLMClient, LLMError
from .voting import VotingMixin


class WorkerPool(VotingMixin):
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

    WorkerPool 继承 VotingMixin：投票相关的 vote / select_best / _run_ballot /
    render_answer 均在 VotingMixin 中定义，这里专注派工与健康逻辑。
    """

    def __init__(self, cfg: Config, exclude: str | None = None,
                 health: ModelHealth | None = None):
        self.cfg = cfg
        self.profiles: dict[str, AgentProfile] = {
            p.name: p for p in cfg.agent_profiles if p.name != exclude
        }
        # 模型健康档案：默认按配置落盘（data/model_health.json），测试可注入独立实例。
        # 派工失败时据此做当日隔离（当天不再派工给终态失败的模型）。
        # 默认构造走 get_health()：同一档案路径全进程共享一个实例，避免多个工人池各持
        # 独立内存账本互相覆盖写盘、且彼此看不到隔离结果（见 health.get_health）。
        self.health = health if health is not None else get_health(cfg.model_health_path)
        # 下线改写需要模型名与清单变量名（如 NODE_A_MODELS），一次性建好映射
        self._model_of = {n: p.model for n, p in self.profiles.items()}
        self._env_of = {n: p.models_env for n, p in self.profiles.items()}
        # 节点健康：连续失败达阈值进入冷却，冷却期内跳过该节点。
        # 冷却时长指数上升，成功即清零。
        self._cooldowns: dict[str, float] = {}  # name -> 冷却到期时间戳
        self._fail_streak: dict[str, int] = {}  # name -> 连续失败次数
        self._cooldown_threshold = self.cfg.as_int(
            self.cfg.collab_cfg.get("cooldown_fails", 2),
            "collaboration.cooldown_fails", 2, minimum=1,
        )
        self._cooldown_base = self.cfg.as_float(
            self.cfg.collab_cfg.get("cooldown_base", 10),
            "collaboration.cooldown_base", 10.0, minimum=0.0,
        )
        # 并发安全：ask_many/vote 用线程池并发调用 ask，健康状态与客户端缓存
        # （_clients）都需加锁保护。（历史上此处还有 LRU 答案缓存与并发在飞去重，
        # 已随「单一 bridge 路线」简化移除——外部主按需发话，命中率极低。）
        self._lock = threading.Lock()
        # 按 profile.name 缓存 LLMClient，避免每次派工都重建连接池
        self._clients: dict[str, LLMClient] = {}
        # pick() 轮转游标（#6）：随默认批量派工的调用推进，把负载摊到整个池，
        # 而非每次死取头部 N 个。单进程一池共享 → 全局 round-robin。
        # _pick_lock 单独一把：MCP 外壳下宿主可并发发工具调用（同步工具跑在线程池），
        # 多个默认派工会同时推进游标。绝不能复用 self._lock——pick()→_skip_reason()
        # →_is_in_cooldown() 已持 self._lock，重入非重入锁会自死锁。
        self._pick_cursor = 0
        self._pick_lock = threading.Lock()

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

        可选 `tags`（逗号分隔）：只保留能力标签**全部命中**的工人，供外部主按任务类型选路
        （标签见 list_agents / health 快照）。缺省 None 不过滤。
        """
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
        """
        now = time.time()
        out: list[dict] = []
        for name, p in self.profiles.items():
            with self._lock:
                until = self._cooldowns.get(name, 0.0)
            quarantined = self.health.quarantined(name)
            out.append({
                "worker": name, "model": p.model,
                "tags": getattr(p, "tags", ""),
                "available": until <= now and not quarantined,
                "cooldown_s": round(max(0.0, until - now), 1),
                "quarantined_today": quarantined,
            })
        return out

    # ---------- 兜底单模型纳入健康体系（#8）----------
    # ask 不带 agent 时走 cfg.llm 的兜底模型，它不是池里的 profile，历史上完全
    # 游离在冷却/当日隔离之外，与主路线容错语义割裂。这里给它一个合成键 solo:<model>，
    # 复用同一套 _skip_reason / _record_result，让兜底路径也"抖了会退避、终态失败当天不再打"。

    def solo_key(self) -> str:
        """兜底单模型在健康体系里的合成键。"""
        return f"solo:{self.cfg.llm.get('model', '')}"

    def is_dispatchable(self, name: str) -> tuple[bool, str]:
        """任意键（含 solo 键）当前能否派工，返回 (可用?, 原因或空串)。"""
        reason = self._skip_reason(name)
        return (reason is None, reason or "")

    def note_result(self, name: str, ok: bool, fatal: bool = False) -> None:
        """记录任意键一次成败：冷却照记；fatal（终态不可重试）才当日隔离。"""
        self._record_result(name, ok=ok, fatal=fatal)

    def _is_in_cooldown(self, name: str) -> bool:
        """该节点是否处于冷却期（冷却期跳过，避免反复打一个坏节点）。"""
        with self._lock:
            until = self._cooldowns.get(name, 0.0)
        return time.time() < until

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
        - 任何失败都累计连击、必要时进**短时冷却**（base×连击，成功即清零、会自动恢复）；
        - 只有 **fatal=True（不可重试的终态错误：认证失败、非重试类 4xx 等）** 才记入
          持久健康档案（ModelHealth）→ **当天不再向该模型派工**。

        为什么这么分：免费中转站天天 timeout/429/5xx/连接抖动是常态，若"任何一次失败即
        全天隔离"，早上被限一次流就把整池打空、当天再不回来。可重试类失败交给冷却做
        短时退避即可，绝不该升级成全天封禁；只有"重试也没用"的终态错误才值得整天拉黑。

        档案落盘在池锁外做，失败路径不在热路径上，多一次小文件写无碍。
        """
        now = time.time()
        with self._lock:
            if ok:
                self._cooldowns.pop(name, None)
                self._fail_streak[name] = 0
                return
            streak = self._fail_streak.get(name, 0) + 1
            self._fail_streak[name] = streak
            if streak >= self._cooldown_threshold:
                # 连续失败越多，冷却越久：base×streak
                self._cooldowns[name] = now + self._cooldown_base * streak
        if not fatal:
            return  # 可重试的瞬时失败：只进冷却，不做当日隔离
        try:
            notice = self.health.record_failure(
                name, self._model_of.get(name, ""), self._env_of.get(name, "")
            )
        except Exception:  # noqa: BLE001 - 健康档案异常绝不影响派工主流程
            notice = None
        if notice:
            print(notice, file=sys.stderr)  # stderr：bridge/--stream 的 stdout 必须纯净

    def ask_result(self, worker: str, prompt: str, system: str | None = None) -> dict:
        """派工核心：返回结构化结果 {worker, ok, status, answer, error}（永不抛异常）。

        全部派工路径（ask / ask_many / collect / vote / run_review）的唯一事实源：
        成败由控制流决定（是否跳过、chat 是否抛错），而不是事后解析文本前缀（#4）。
        跳过判定（冷却 / 当日隔离）也在这里直接算出，批量路径无需再预过滤或嗅探。
        """
        p = self.profiles.get(worker)
        if not p:
            return {"worker": worker, "ok": False, "status": ST_MISSING, "answer": "",
                    "error": f"子智能体 '{worker}' 不存在。可用：{', '.join(self.profiles) or '无'}"}
        reason = self._skip_reason(worker)
        if reason == "冷却中":
            return {"worker": worker, "ok": False, "status": ST_COOLDOWN, "answer": "",
                    "error": f"子智能体 '{worker}' 近期连续失败，正在冷却中，建议稍后再试或换其它工人"}
        if reason == "当日失败隔离":
            return {"worker": worker, "ok": False, "status": ST_QUARANTINED, "answer": "",
                    "error": (f"子智能体 '{worker}' 今日调用已失败，当天隔离不再派工，"
                              "请换其它工人或明天再试")}
        return self._call_result(p, system, prompt)

    def ask(self, worker: str, prompt: str, system: str | None = None) -> str:
        """派一个子任务给单个工人，返回其回答文本（含失败说明，不抛异常）。

        薄封装：结构化判定交给 ask_result，这里只渲染成人读文本（向后兼容）。
        免费节点友好：节点在**短时冷却**期、或当天已出现**终态不可重试错误**（当日隔离）
        时直接快速跳过并提示，不傻等；冷却到期自动回来，隔离只针对当天的坏模型。
        """
        return self.render_answer(self.ask_result(worker, prompt, system))

    def _call_result(self, p: AgentProfile, system: str | None, prompt: str) -> dict:
        """真正打一次网络：取（或复用）LLMClient → chat → 按控制流记结构化成败。

        失败按性质分级记账（#1/#2）：LLMError 且 retryable=False（认证失败、非重试 4xx 等
        "重试也没用"的终态错误）→ fatal=True，触发当日隔离；其余失败（可重试错误耗尽、
        意外异常、连取 client 都失败）→ fatal=False，只进短时冷却、当天仍可回来。
        无论成败都返回 dict、不抛异常；ask_many/collect/vote 依赖该契约。
        """
        try:
            client = self._client_for(p)
            messages = [
                {"role": "system", "content": system or WORKER_SYSTEM},
                {"role": "user", "content": prompt},
            ]
            try:
                resp = client.chat(messages)
            except LLMError as e:
                # 不可重试的终态错误才判"当天别碰"；可重试类只冷却
                self._record_result(p.name, ok=False, fatal=not e.retryable)
                return {"worker": p.name, "ok": False, "status": ST_ERROR, "answer": "",
                        "error": f"调用子智能体 '{p.name}' 失败 [{e.error_type}]：{e}"}
            except Exception as e:  # noqa: BLE001 - 意外异常按可重试处理，别全天拉黑
                self._record_result(p.name, ok=False, fatal=False)
                return {"worker": p.name, "ok": False, "status": ST_ERROR, "answer": "",
                        "error": f"调用子智能体 '{p.name}' 失败：{type(e).__name__}: {e}"}
            self._record_result(p.name, ok=True)
            return {"worker": p.name, "ok": True, "status": ST_OK,
                    "answer": resp.get("content", "") or "(空回复)", "error": ""}
        except Exception as e:  # noqa: BLE001 - 连取 client 都失败，也不让派工崩（只冷却）
            self._record_result(p.name, ok=False, fatal=False)
            return {"worker": p.name, "ok": False, "status": ST_ERROR, "answer": "",
                    "error": f"调用子智能体 '{p.name}' 异常：{type(e).__name__}: {e}"}

    def _client_for(self, p: AgentProfile) -> LLMClient:
        """按 profile.name 复用 LLMClient；首次访问时按工人配置创建并缓存。
        timeout / max_retries 从全局配置读取，保持与主智能体一致的重试策略。

        加锁创建：并发派工（ask_many/vote）下同一工人会被多个线程同时首次访问，
        不加锁会各建一个 LLMClient（多套连接池，其中一个被覆盖后无人引用）。
        调用方（_lead）在调本方法时并未持有 self._lock，故不会自锁。
        """
        with self._lock:
            client = self._clients.get(p.name)
            if client is not None:
                return client
            timeout = self.cfg.as_float(
                self.cfg.llm_cfg.get("timeout", 120), "llm.timeout", 120.0, minimum=1.0
            )
            max_retries = self.cfg.as_int(
                self.cfg.llm_cfg.get("max_retries", 2), "llm.max_retries", 2, minimum=0
            )
            client = LLMClient(
                p.base_url, p.api_key, p.model,
                timeout=timeout, max_retries=max_retries,
            )
            self._clients[p.name] = client
            return client

    def close(self) -> None:
        """释放全部工人客户端连接池（幂等）。池被弃用时由持有方调用。"""
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for c in clients:
            try:
                c.close()
            except Exception:  # noqa: BLE001 - 释放失败不应影响上层收尾
                pass

    # ---------- 并行派工 ----------

    def _wait_gather(
        self, futs: dict[Future, str], timeout: float, on_done=None,
    ) -> dict[str, dict | None]:
        """等待一批 {Future: 工人名}，整体限时，返回 {工人: 结果 dict 或 None}。

        每个值是 ask_result 的结构化 dict；None 表示整体限时到点仍未完成（调用方据此
        标 timeout）。用 `wait(..., FIRST_COMPLETED)` 边完成边收集；`on_done(worker, res)`
        每有一个工人回来就回调一次（ask_many 用它记耗时）。回调异常被吞。
        ask_result 本身不抛，这里的 except 只是兜底：万一抛了也归一成 error dict。
        """
        fut_to_worker = futs
        deadline = time.time() + timeout
        out: dict[str, dict | None] = {}
        pending = set(fut_to_worker)
        while pending:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            done, pending = wait(
                pending, timeout=min(remaining, 0.5),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                w = fut_to_worker[fut]
                try:
                    res = fut.result()
                except Exception as e:  # noqa: BLE001 - 兜底归一成结构化 error，绝不下游解析文本
                    res = {"worker": w, "ok": False, "status": ST_ERROR, "answer": "",
                           "error": f"派工异常：{type(e).__name__}: {e}"}
                out[w] = res
                if on_done is not None:
                    try:
                        on_done(w, res)
                    except Exception:  # noqa: BLE001 - 回调异常不影响派工
                        pass
        for fut in pending:  # 超时：放弃等待（已启动的后台自行收尾）
            fut.cancel()
            out[fut_to_worker[fut]] = None
        return out

    def _effective_timeout(self, n: int, timeout: float | None) -> float:
        """批量派工的整体限时：显式传入优先，缺省按「批次 × 单请求超时」推导。

        为什么不用固定 300s：批次 = ceil(n / max_workers)，每批最长 llm.timeout 秒。
        池大 + 并发低时（实测 25 个工人 / 3 并发 = 9 批），固定限时在单工人耗时超过
        「限时 ÷ 批次」秒时（>33s 就触发）会把后面所有批次判为"未完成"丢弃 ——
        票和草稿都拿不全，额度却已经花掉。自适应后限时随规模伸缩，并加 30s 余量
        给排队与序列化开销；配合 pick() 的参与人数上限，规模本身也回到可控区间。
        """
        if timeout is not None:
            return float(timeout)
        per_call = self.cfg.as_float(
            self.cfg.llm_cfg.get("timeout", 120) or 120, "llm.timeout", 120.0, minimum=1.0
        )
        batches = max(1, -(-max(1, n) // self.cfg.max_workers))
        return batches * per_call + 30.0

    def _spawn(self, fn, sem: threading.Semaphore) -> Future:
        """起一个 daemon 工作线程执行 fn，返回其 Future（取代 ThreadPoolExecutor）。

        为什么不用 ThreadPoolExecutor（审计 P2）：它的 worker 是非 daemon 线程，
        解释器退出时 atexit 会把它们全部 join——批量派工里的「放弃等待」只是不再
        等结果，bridge/mcp 进程收尾仍会被在飞请求（最长 LLM timeout×retries）拖住。
        daemon 线程让退出干脆。取消语义保留：整体限时到点后尚未开跑的任务被 cancel，
        set_running_or_notify_cancel 返回 False ⇒ 不再打网络，不白烧免费额度。
        """
        fut: Future = Future()

        def _run() -> None:
            with sem:
                if not fut.set_running_or_notify_cancel():
                    return  # 已被放弃（cancel），不派工
                try:
                    fut.set_result(fn())
                except BaseException as e:  # noqa: BLE001 - 异常必须进 future，不能杀线程
                    fut.set_exception(e)

        threading.Thread(target=_run, daemon=True).start()
        return fut

    def _run_parallel(
        self, workers: list[str], prompt: str, system: str | None, timeout: float,
        on_done=None,
    ) -> dict[str, dict | None]:
        """并行派工给多个工人并限时收集；未完成的放弃等待（daemon 线程，不拖退出）。

        并发单元固定为 ask_result（结构化、不抛异常），也是各类 spy 测试的 patch 点。
        并发上限由信号量把守（与原线程池容量等价），批次节奏与 _effective_timeout
        的推导保持一致。on_done 透传给 _wait_gather（ask_many 用它记每个工人耗时；
        collect/vote 不传）。
        """
        if not workers:
            return {}
        sem = threading.Semaphore(min(len(workers), self.cfg.max_workers))
        futs = {
            self._spawn(lambda w=w: self.ask_result(w, prompt, system), sem): w
            for w in workers
        }
        return self._wait_gather(futs, timeout, on_done=on_done)

    def collect(
        self, workers: list[str], prompt: str, system: str | None = None,
        timeout: float | None = None,
    ) -> list[tuple[str, str]]:
        """并行收集多个工人对同一任务的成功回答，按入参工人顺序返回（确定性）。

        与 ask_many 的区别：只返回成功回答的 (工人, 回答) 列表，且顺序与
        workers 入参顺序一致，不随完成先后漂移——投票编号、候选顺序
        依赖该确定性（as_completed 完成序会导致同一任务两次运行编号互换）。
        失败/冷却/当日隔离/超时/空回复的工人不进结果。

        这里**不做**冷却/隔离预过滤：判定交给 ask_result 内部，返回结构化 status，
        这里只按 status==ST_OK 过滤——成败由控制流决定，绝不解析回答文本前缀（#4）。
        """
        valid = [w for w in dict.fromkeys(workers) if w in self.profiles]
        if not valid:
            return []
        got = self._run_parallel(
            valid, prompt, system, self._effective_timeout(len(valid), timeout)
        )
        return [
            (w, got[w]["answer"])
            for w in valid
            if got.get(w) is not None and got[w]["status"] == ST_OK and got[w]["answer"]
        ]

    def _gather_many(
        self, workers: list[str], prompt: str, system: str | None, timeout: float | None,
    ) -> dict:
        """并行派工的收集核心：ask_many（文本）与 ask_many_structured（结构化）共用。

        返回 {valid, got, elapsed, limit}。got[worker] 为 None 表示超时未完成；
        否则是 ask_result 的结构化 dict。elapsed[worker] 为该工人从派工到回来的毫秒数。

        不认识的工人名**不再静默丢弃**（审计 P1）：照样进入结果、由 ask_result 如实
        带回 status=missing——bridge 文档承诺的 status 词表含 missing，且外部主点名
        拼错时若只收到「全部子 AI 均未成功」，会拿着同一个错名字永远重试。
        """
        valid = list(dict.fromkeys(workers))
        if not valid:
            return {"valid": [], "got": {}, "elapsed": {}, "limit": 0.0}
        limit = self._effective_timeout(len(valid), timeout)
        t0 = time.monotonic()
        elapsed: dict[str, int] = {}

        def _on_done(w: str, res: dict | None) -> None:
            # res 来自 _wait_gather 的 fut.result()：ask_result 的结构化 dict 或 None（超时）
            elapsed[w] = int((time.monotonic() - t0) * 1000)

        got = self._run_parallel(valid, prompt, system, limit, on_done=_on_done)
        return {"valid": valid, "got": got, "elapsed": elapsed, "limit": limit}

    def ask_many(
        self, workers: list[str], prompt: str, system: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """同一子任务并行派给多个工人，收集全部回答（失败也如实带回）。

        输出按入参工人顺序排列（确定性）；timeout 对收集阶段整体限时，缺省按
        参与人数与并发上限自适应（见 _effective_timeout）。超时工人以「未完成」带回。
        冷却/当日隔离工人不预过滤，交给 ask_result 内部判定并如实带回跳过说明。
        """
        r = self._gather_many(workers, prompt, system, timeout)
        valid = r["valid"]
        if not valid:
            return f"错误：没有可用的子智能体。可用：{', '.join(self.profiles) or '无'}"
        got, limit = r["got"], r["limit"]
        blocks: list[str] = []
        for w in valid:  # 按入参顺序输出，含跳过项
            res = got.get(w)
            if res is None:  # 整体限时到点仍未完成
                body = f"失败：超过 {limit:g}s 未完成，已放弃等待（后台线程会自行收尾）"
            elif res["status"] in (ST_COOLDOWN, ST_QUARANTINED):
                label = "冷却中" if res["status"] == ST_COOLDOWN else "当日隔离"
                body = f"错误：工人{label}，本轮跳过"
            else:  # ok 或 error：由结构化 status 渲染成人读文本，绝不回头解析文本
                body = self.render_answer(res)
            blocks.append(f"### 工人 {w} 的结果\n{body}")
        return "\n\n".join(blocks)

    def ask_many_structured(
        self, workers: list[str], prompt: str, system: str | None = None,
        timeout: float | None = None,
    ) -> list[dict]:
        """并行派工的结构化结果（外部主程序化消费，替代 Markdown 大块文本）。

        返回按入参顺序的 list[dict]，每项 {worker, ok(bool), status, answer, elapsed_ms}。
        status ∈ {ok, error, cooldown, quarantined, missing, timeout}（与 ask_result 同一套
        稳定码，不再翻译成中文）：直接取自结构化判定（#4，不解析文本前缀），ok 只认
        status==ok，answer 仅在成功时为正文、其余为空串。
        """
        r = self._gather_many(workers, prompt, system, timeout)
        got, elapsed = r["got"], r["elapsed"]
        out: list[dict] = []
        for w in r["valid"]:
            res = got.get(w)
            if res is None:  # 整体限时到点仍未完成
                out.append({"worker": w, "ok": False, "status": ST_TIMEOUT,
                            "answer": "", "elapsed_ms": elapsed.get(w)})
                continue
            out.append({
                "worker": w, "ok": res["status"] == ST_OK,
                "status": res["status"],
                "answer": res["answer"] if res["status"] == ST_OK else "",
                "elapsed_ms": elapsed.get(w),
            })
        return out
