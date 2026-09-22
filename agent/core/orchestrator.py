"""多智能体编排：工人池（子智能体）与派工工具。

核心理念（三个臭皮匠顶个诸葛亮）：
- 主智能体：拆解、派工、汇总、把关（它自己也可以是个免费弱模型）
- 工人池：智能体池里除主之外的全部成员，纯文本执行器（不挂本地工具）
- swarm 模式：把 ask_worker / ask_workers 做成主智能体的工具，
  由主智能体在 Agent Loop 里自主决定何时拆解、派工、汇总。
- M5 增强：ask 结果 LRU 缓存（同工同题不重复烧钱）；vote 两步投票
  （先收集各方方案，再让全体对编号方案投票，超阈值即共识）。
"""
from __future__ import annotations

import concurrent.futures
import re
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait

from ..config import AgentProfile, Config
from .health import ModelHealth, get_health
from .llm import LLMClient, LLMError

WORKER_SYSTEM = (
    "你是协作团队中的工人智能体。认真完成分配给你的子任务，"
    "直接给出结果内容本身，不要客套、不要复述任务。"
    "如果子任务无法完成，明确说明原因和你的困难。"
)
VOTER_SYSTEM = (
    "你是协作团队中的评审工人。下面给你一份方案清单，"
    "请选出你认为最好的那份，只输出它的编号（一个数字），不要输出任何其他内容。"
)

# ---------- 派工结果状态（#4：结构化取代对「错误：」文本前缀的字符串嗅探）----------
# 成败由**控制流**决定（是否跳过、chat 是否抛错），绝不从工人回答文本里猜：
# 工人正常回答若以「错误：」开头也不会被误判成失败。
ST_OK = "ok"                    # 成功拿到回答
ST_ERROR = "error"              # 打了网络但失败（LLMError / 意外异常 / 取 client 失败）
ST_COOLDOWN = "cooldown"        # 冷却中，本轮跳过
ST_QUARANTINED = "quarantined"  # 当日失败隔离，本轮跳过
ST_MISSING = "missing"          # 指定工人不在池里
ST_TIMEOUT = "timeout"          # 批量派工整体限时内未回来（由收集层标注）
# 以上状态码即对外 status 词表（ask_result / ask_many_structured / 单 ask 统一使用），
# 不再另设中文标签翻译层：外部主按稳定英文码程序化消费。
# 批量派工的整体限时不再是固定常量：按「批次 × 单请求超时」自适应推导，
# 见 WorkerPool._effective_timeout —— 固定值在「池大 + 并发低」时必然截断尾部批次。
# 单请求超时直接取 llm.timeout，与主智能体、工人客户端同源。


class WorkerPool:
    """子智能体池：纯文本执行器（不挂本地工具，兼容性最好、最安全）。

    免费节点友好：单节点抖动/限流/不可用不拖垮整体——连续失败进入**短时冷却**，
    冷却中跳过该节点（ask 直接返回冷却提示），其它节点照常工作，冷却到期后自动回来。
    健康档案（ModelHealth）叠加在冷却之上：仅当某模型出现**终态不可重试错误**
    （认证失败、非重试 4xx）才当天隔离（当天不再派工）；可重试的瞬时失败只进冷却，
    不会升级成全天封禁——见 _record_result。
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
        # 按 profile.name 缓存 LLMClient，避免每次派工都重建连接池（任务3）
        self._clients: dict[str, LLMClient] = {}
        # pick() 轮转游标（#6）：随默认批量派工的调用推进，把负载摊到整个池，
        # 而非每次死取头部 N 个。单进程一池共享 → 全局 round-robin。
        self._pick_cursor = 0

    # ---------- 基础派工 ----------

    def names(self) -> list[str]:
        return list(self.profiles)

    def pick(self, limit: int | None = None, tags: str | None = None) -> list[str]:
        """挑出本次批量派工要用的工人：跳过冷却与当日隔离，跨调用轮转分摊负载。

        缺省上限取 collaboration.max_participants（<=0 表示不限，返回全部可用）。

        为什么轮转（#6）：池子「一站多模型」轻易几十个，而固定取头部 N 个会让同一批
        头部节点反复挨打、最快被限流/隔离，尾部几十个模型整天闲置——既没摊薄免费额度
        又放大单点抖动。用一个随调用推进的游标做 round-robin：
        - 单次调用内仍是**连续窗口**（确定性）——投票编号、流水线候选顺序不会在一次
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
        start = self._pick_cursor % n
        self._pick_cursor += min(cap, n)
        window = live[start:] + live[:start]  # 轮转后的连续窗口，仍保持池内相对顺序
        return window[:cap]

    def overview(self) -> str:
        if not self.profiles:
            return "（当前池里没有其它智能体，所有工作由你自己完成）"
        # 当日隔离直接标注在清单里：主智能体选工人时就避开，不用挨个撞错误提示
        return "\n".join(
            f"- {p.name}: model={p.model}"
            + (f" tags={p.tags}" if getattr(p, "tags", "") else "")
            + ("（今日调用已失败，当天隔离中，勿派工）" if self.health.quarantined(p.name) else "")
            + (f"（{p.note}）" if p.note else "")
            for p in self.profiles.values()
        )

    def health_status(self) -> list[dict]:
        """每个工人的当前可用性快照，供外部主派工前预检（配合能力标签选路）。

        available=False 表示当前不可派：冷却中（cooldown_s 剩余秒）或当日失败隔离。
        缓存命中不受此限（ask 里先查缓存），这里只是给主的"该不该派"参考。
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

    @staticmethod
    def render_answer(r: dict) -> str:
        """把结构化派工结果渲染成对外文本（单 ask / ask_many 的人读输出）。

        成败只认 status，绝不回头解析回答文本里的「错误：」字样——单一事实源：
        ask()/ask_many 的人读文本都由这里生成，内部判定一律走 ask_result 的结构。
        """
        if r["status"] == ST_OK:
            return r["answer"]
        return f"错误：{r['error']}"

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

    def _run_parallel(
        self, workers: list[str], prompt: str, system: str | None, timeout: float,
        on_done=None,
    ) -> dict[str, dict | None]:
        """并行派工给多个工人并限时收集，调用方拿到后 shutdown(wait=False) 不阻塞。

        并发单元固定为 ask_result（结构化、不抛异常），也是各类 spy 测试的 patch 点。
        on_done 透传给 _wait_gather（ask_many 用它记每个工人耗时；collect/vote 不传）。
        """
        ex = ThreadPoolExecutor(max_workers=min(len(workers), self.cfg.max_workers))
        try:
            futs = {ex.submit(self.ask_result, w, prompt, system): w for w in workers}
            return self._wait_gather(futs, timeout, on_done=on_done)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    def collect(
        self, workers: list[str], prompt: str, system: str | None = None,
        timeout: float | None = None,
    ) -> list[tuple[str, str]]:
        """并行收集多个工人对同一任务的成功回答，按入参工人顺序返回（确定性）。

        与 ask_many 的区别：只返回成功回答的 (工人, 回答) 列表，且顺序与
        workers 入参顺序一致，不随完成先后漂移——投票编号、流水线候选
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
        """
        valid = [w for w in dict.fromkeys(workers) if w in self.profiles]
        if not valid:
            return {"valid": [], "got": {}, "elapsed": {}, "limit": 0.0}
        limit = self._effective_timeout(len(valid), timeout)
        t0 = time.monotonic()
        elapsed: dict[str, int] = {}

        def _on_done(w: str, res: str | None) -> None:
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

    # ---------- 两步投票（M5：多人投票取共识） ----------

    def _run_ballot(
        self, candidates: list[tuple[str, str]], voters: list[str], timeout: float | None = None
    ) -> tuple[int, dict[int, int], int, int, int] | None:
        """对编号候选方案发起投票（两步投票的第二步）。

        返回 (best_idx, tally, total, best_n, invalid)：total 是**有效票数**，
        invalid 是投了不存在编号的废票数（调用方据此如实提示）。
        候选编号与 candidates 下标一一对应（candidates 顺序由调用方保证确定性）。
        无任何可解析选票（超时/工人都没输出编号）返回 None。
        阈值不在这里判（共识与否由调用方按得票比例决定），故本函数不收阈值参数。
        """
        numbered = [
            f"[{i}] {w}\n预览：{ans[:80]}".replace("\n", " ")
            for i, (w, ans) in enumerate(candidates, 1)
        ]
        ballot = "候选方案：\n" + "\n".join(numbered) + "\n\n请只输出你认可方案的编号数字。"
        got = self._run_parallel(
            voters, ballot, VOTER_SYSTEM, self._effective_timeout(len(voters), timeout)
        )
        votes: list[int] = []
        for res in got.values():
            # 只有成功派工的 ST_OK 回答才算"投票意见"：错误/跳过/超时按结构化 status
            # 排除，绝不解析文本前缀（#4）。ask_result 已保证失败时 answer="" —— 双重保险，
            # 冷却提示、HTTP 状态码、"超过 Ns 未完成"里带的数字都不会被误当成选票。
            if res is None or res["status"] != ST_OK:
                continue
            m = re.search(r"(?<!\d)\d+(?!\d)", res["answer"])
            if m:
                votes.append(int(m.group()))
        if not votes:
            return None
        # 分母只算**有效票**：模型偶尔会随口给个不存在的编号（如候选只有 3 份却投 9），
        # 这种废票不该出现在共识比例的分母里。原实现 total = len(votes) 把废票也算进
        # 分母，会出现"有效票 1/1 全投方案一、却被另 2 张废票稀释成 1/3 判为未达共识"
        # 的错误结论 —— 投票是"取共识"的关键路径，分母错就等于结论错。
        valid = [v for v in votes if 1 <= v <= len(candidates)]
        if not valid:
            return None
        total = len(valid)
        invalid = len(votes) - total
        tally: dict[int, int] = {}
        for v in valid:
            tally[v] = tally.get(v, 0) + 1
        best_idx, best_n = max(tally.items(), key=lambda kv: kv[1])
        return best_idx, tally, total, best_n, invalid

    def select_best(
        self,
        candidates: list[tuple[str, str]],
        voters: list[str] | None = None,
        threshold: float | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str, str] | None:
        """对已有候选方案投票择优（复用两步投票的第二步，不重复收集阶段）。

        返回 (胜出工人名, 胜出全文, 票况摘要)；无解析选票返回 None。
        供流水线修订阶段等「候选已就绪」的场景复用。voters 缺省取 pick() 的
        可用工人子集（受 collaboration.max_participants 约束），不再默认全池。
        """
        th = threshold if threshold is not None else self.cfg.as_float(
            self.cfg.collab_cfg.get("vote_threshold", 0.5),
            "collaboration.vote_threshold", 0.5, minimum=0.0,
        )
        pool = [w for w in dict.fromkeys(voters or self.pick()) if w in self.profiles]
        if not pool:
            return None
        result = self._run_ballot(candidates, pool, timeout)
        if result is None:
            return None
        best_idx, tally, total, best_n, _invalid = result
        winner = candidates[best_idx - 1]
        summary = (
            f"投票 {best_n}/{total} 票（阈值 {th:.0%}）；票况："
            + ", ".join(f"[{i}]={tally.get(i, 0)}" for i in range(1, len(candidates) + 1))
        )
        return winner[0], winner[1], summary

    def vote(
        self,
        prompt: str,
        workers: list[str] | None = None,
        threshold: float | None = None,
        timeout: float | None = None,
    ) -> str:
        """两步投票：先让每个工人给出方案，再让全体工人对方案编号投票。

        投票规则：
        1) 并行收集各工人对同一任务的目标作答（成功结果才进入候选池），
           候选顺序按入参工人顺序排列（编号确定性，重跑不互换）。
        2) 把候选方案编号（附 80 字符预览）重新发给全体工人，每人选一个编号。
        3) 统计得票：最高票数 / 总票数 >= 阈值（默认 collaboration.vote_threshold，
           通常取 0.5，即过半共识）则宣布达成共识，否则列出票况请主智能体裁决。

        workers 缺省取 pick()（按 collaboration.max_participants 限制参与人数、
        跳过冷却与当日隔离的工人）—— 全池投票在「一站多模型」的池子里意味着
        收集 N 次 + 投票 N 次共 2N 个请求，免费额度扛不住且尾部批次必然超时。

        返回结构化文本（含每方案得票与共识结论），供主智能体直接采信或人工复核。
        """
        th = threshold if threshold is not None else self.cfg.as_float(
            self.cfg.collab_cfg.get("vote_threshold", 0.5),
            "collaboration.vote_threshold", 0.5, minimum=0.0,
        )
        pool = [w for w in dict.fromkeys(workers or self.pick()) if w in self.profiles]
        if not pool:
            return "错误：没有可投票的子智能体。"

        # 第一步：收集各工人方案（失败者淘汰，不进候选池；顺序确定性）
        candidates = self.collect(pool, prompt, timeout=timeout)
        if not candidates:
            return "错误：所有工人均未给出可用方案，无法投票。"

        # 第二步：编号后发给全体投票
        result = self._run_ballot(candidates, pool, timeout)
        if result is None:
            return "错误：投票结果无法解析（工人未输出编号）。"
        best_idx, tally, total, best_n, invalid = result
        ratio = best_n / total
        body = ["## 候选方案", *[f"[{i}] {w}：{ans[:96]}…" for i, (w, ans) in enumerate(candidates, 1)]]
        body.append("## 投票结果")
        for i in range(1, len(candidates) + 1):
            body.append(f"方案[{i}]: {tally.get(i, 0)}/{total} 票")
        if invalid:
            body.append(
                f"（另有 {invalid} 张票投了不存在的编号，已忽略；比例按有效票 {total} 张计算）"
            )
        if ratio >= th:
            winner = candidates[best_idx - 1]
            body.append(f"## 共识达成（{best_n}/{total} 票 ≥ {th:.0%}）")
            body.append(f"胜出方案（{winner[0]}）：\n{winner[1]}")
        else:
            body.append(f"## 未达共识（最高 {best_n}/{total}，阈值 {th:.0%}）")
            body.append("请主智能体结合候选内容自行裁决或要求重投。")
        return "\n".join(body)

