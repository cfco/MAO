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
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, wait

from ..config import AgentProfile, Config
from ..tools.base import FunctionTool, ToolRegistry
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
# 批量派工的整体限时不再是固定常量：按「批次 × 单请求超时」自适应推导，
# 见 WorkerPool._effective_timeout —— 固定值在「池大 + 并发低」时必然截断尾部批次。
# 单请求超时直接取 llm.timeout，与主智能体、工人客户端同源。


class WorkerPool:
    """子智能体池：纯文本执行器（不挂本地工具，兼容性最好、最安全）。

    免费节点友好：单节点抖动/限流/不可用不拖垮整体——连续失败进入冷却期，
    冷却中跳过该节点（ask 直接返回冷却提示），其它节点照常工作。
    健康档案（ModelHealth）叠加在冷却之上：某模型当天出现终态失败，当天不再
    派工（缓存命中不受影响）；连续多个"运行日"失败达到阈值自动在 .env.example 标 # 下线。
    """

    def __init__(self, cfg: Config, exclude: str | None = None,
                 health: ModelHealth | None = None):
        self.cfg = cfg
        self.profiles: dict[str, AgentProfile] = {
            p.name: p for p in cfg.agent_profiles if p.name != exclude
        }
        # 模型健康档案：默认按配置落盘（data/model_health.json），测试可注入独立实例。
        # 派工失败时据此做当日隔离；连续失败达 retire_days 天自动改写 .env.example 下线。
        # 默认构造走 get_health()：同一档案路径全进程共享一个实例——WorkerPool 是
        # "每个 Agent（=Web 每个会话）一份"，各持一份独立账本会互相覆盖写盘、
        # 且彼此看不到隔离结果（见 health.get_health 的详细说明）。
        self.health = health if health is not None else get_health(
            cfg.model_health_path, cfg.env_example_path, retire_days=cfg.health_retire_days,
        )
        # 下线改写需要模型名与清单变量名（如 NODE_A_MODELS），一次性建好映射
        self._model_of = {n: p.model for n, p in self.profiles.items()}
        self._env_of = {n: p.models_env for n, p in self.profiles.items()}
        # 工人回答 LRU 缓存：同(工人,任务,角色)不重复调用，省额度和时延。
        # 只缓存成功回答，失败结果不缓存（便于下次重试）。
        # cache_size=0 → 关闭缓存（外部主按需发话、prompt 几乎不逐字重复时，命中率
        # 极低，关掉省去内存与"陈旧答案被复用"的心智负担）。
        self._cache: OrderedDict[tuple, str] = OrderedDict()
        self._cache_size = self.cfg.as_int(
            self.cfg.collab_cfg.get("cache_size", 32), "collaboration.cache_size", 32, minimum=0
        )
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
        # 并发安全：ask_many/vote 用线程池并发调用 ask，缓存、健康状态与
        # 客户端缓存（_clients）都需加锁保护
        self._lock = threading.Lock()
        # 在飞去重：key -> Future，并发同 key 未命中缓存时只打一次网络（任务：防抖）
        self._inflight: dict[tuple, Future] = {}
        # 在飞去重兜底超时（秒）：发起者超过该时长未返回，跟随者接管重新打网络，
        # 避免发起者挂死/超长导致跟随者线程无限阻塞（ask_many/vote 的外层超时只能中断
        # 调用方，内部跟随者线程仍会卡在 fut.result() 造成线程泄漏）。默认 120s。
        self._inflight_timeout = self.cfg.as_float(
            self.cfg.collab_cfg.get("inflight_timeout", 120.0),
            "collaboration.inflight_timeout", 120.0, minimum=0.0,
        )
        # 按 profile.name 缓存 LLMClient，避免每次派工都重建连接池（任务3）
        self._clients: dict[str, LLMClient] = {}

    # ---------- 基础派工 ----------

    def names(self) -> list[str]:
        return list(self.profiles)

    def pick(self, limit: int | None = None) -> list[str]:
        """挑出本次批量派工要用的工人：按池内确定性顺序，跳过冷却与当日隔离。

        缺省上限取 collaboration.max_participants（<=0 表示不限，返回全部可用）。
        为什么需要上限：池子按「一站多模型」组织时轻易几十个模型，而并发上限
        max_workers 只有个位数 —— 全池派工会一次打出几十个请求，尾部批次撞上
        整体限时被判"未完成"丢弃。顺序保持确定性，投票编号与流水线候选才不会
        随运行漂移。
        """
        cap = self.cfg.max_participants if limit is None else limit
        live = [w for w in self.profiles if self._skip_reason(w) is None]
        if cap is None or cap <= 0:
            return live
        return live[:cap]

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

    def _record_result(self, name: str, ok: bool) -> None:
        """更新节点健康状态：失败累计，达阈值进冷却（时长指数上升）；成功清零。

        失败同时记入持久健康档案（ModelHealth）：当天不再派工；连续失败达
        retire_days 个"运行日"（程序实际启动过的天）则在 .env.example 标 # 下线。
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
        try:
            notice = self.health.record_failure(
                name, self._model_of.get(name, ""), self._env_of.get(name, "")
            )
        except Exception:  # noqa: BLE001 - 健康档案异常绝不影响派工主流程
            notice = None
        if notice:
            print(notice, file=sys.stderr)  # stderr：bridge/--stream 的 stdout 必须纯净

    @staticmethod
    def _is_error(out: str) -> bool:
        """判断工人回答是否为错误。
        只认明确的错误前缀，避免正常回复被误判（比如以"工具列表"开头的正常回答）。
        """
        return out.startswith("错误：") or out.startswith("失败：")

    # ask 在「冷却中」/「当日隔离」时的固定标记词，供批量路径识别"跳过"（见 _skip_kind）
    _SKIP_MARKS = {
        "正在冷却中": "冷却中",
        "当天隔离不再派工": "当日隔离",
    }

    @classmethod
    def _skip_kind(cls, out: str) -> str | None:
        """识别 ask 返回的是否为"跳过类"提示（冷却/当日隔离），命中返回原因文案。

        批量路径（ask_many）不再预过滤工人，改由这里对结果分类。要求同时命中
        「错误：子智能体」前缀与固定标记词，避免把工人正常回答里恰好出现这些
        字样的内容误判成跳过。
        """
        if not out.startswith("错误：子智能体"):
            return None
        for mark, label in cls._SKIP_MARKS.items():
            if mark in out:
                return label
        return None

    def ask(self, worker: str, prompt: str, system: str | None = None) -> str:
        """派一个子任务给单个工人，返回其回答文本（含失败说明，不抛异常）。

        命中缓存直接返回（成功回答才进缓存，刷新 LRU 热度在锁内完成）。
        免费节点友好：节点在冷却期、或当天已出现终态失败（当日隔离）时直接
        快速跳过并提示，不傻等；两者都不挡缓存命中。
        并发去抖：多个线程同时派同一 (工人,任务,角色) 且都未命中缓存时，
        只有第一个线程（发起者）真正打网络，其余线程（跟随者）等待并复用同一结果，
        避免重复烧钱；发起者挂死/超长未返回时，跟随者带超时兜底接管（见 _follow），
        不会无限阻塞。网络调用本身在锁外执行，避免阻塞并发派工。
        """
        p = self.profiles.get(worker)
        if not p:
            return f"错误：子智能体 '{worker}' 不存在。可用：{', '.join(self.profiles) or '无'}"
        # 缓存键归一：system 字段做 strip + None 归一，避免以下情况重复打网络：
        # - 调用者有时传 None、有时传 ""（语义都是"用默认角色"）
        # - 调用者传 "你是工人。" 与 " 你是工人。"（仅前后空白差异）
        sys_norm = (system or "").strip() or None
        key = (worker, prompt, sys_norm)
        with self._lock:  # 缓存命中 + 冷却判定 + 在飞去重需在同一把锁内（并发派工安全）
            if key in self._cache:
                # 缓存命中零成本，不应被冷却/当日隔离挡住：它们的意义是别再打坏节点，
                # 已存下的答案照用。原实现先查冷却后查缓存，误伤命中缓存的请求。
                self._cache.move_to_end(key)
                return self._cache[key]
            if time.time() < self._cooldowns.get(worker, 0.0):
                return f"错误：子智能体 '{worker}' 近期连续失败，正在冷却中，建议稍后再试或换其它工人"
            if self.health.quarantined(worker):
                return (
                    f"错误：子智能体 '{worker}' 今日调用已失败，当天隔离不再派工，"
                    "请换其它工人或明天再试"
                )
            # 并发同 key 去重：已有在飞请求则当前线程作为跟随者等待结果，
            # 不重复打网络；否则成为发起者，建立 Future 后离开锁做网络。
            fut = self._inflight.get(key)
            if fut is not None:
                is_leader = False
            else:
                fut = Future()
                self._inflight[key] = fut
                is_leader = True

        if not is_leader:
            # 跟随者：等待发起者结果；发起者超时/挂死时自动接管，避免无限阻塞（防抖兜底）
            return self._follow(key, fut, worker, prompt, system)

        # 发起者：执行网络调用并结算
        return self._lead(key, fut, p, system, prompt)

    def _lead(self, key: tuple, fut: Future, p: AgentProfile, system: str | None, prompt: str) -> str:
        """发起者路径：打网络 + 结算。

        结算（pop inflight + 写缓存 + set_result）必须在同一把锁内完成，
        保证没有窗口期让新跟随者错过 inflight（否则它会误以为没有 inflight 而成为新发起者）。
        """
        # 取（或复用）该工人的 LLMClient，避免每次派工重建连接池（任务3）
        out = ""
        try:
            client = self._client_for(p)
            messages = [
                {"role": "system", "content": system or WORKER_SYSTEM},
                {"role": "user", "content": prompt},
            ]
            try:
                resp = client.chat(messages)
            except LLMError as e:
                self._record_result(p.name, ok=False)
                out = f"错误：调用子智能体 '{p.name}' 失败 [{e.error_type}]：{e}"
            except Exception as e:  # noqa: BLE001 - 兜底：其它异常也计入失败
                self._record_result(p.name, ok=False)
                out = f"错误：调用子智能体 '{p.name}' 失败：{type(e).__name__}: {e}"
            else:
                self._record_result(p.name, ok=True)  # 正常返回即成功，超时兜底在 ask_many
                out = resp.get("content", "") or "(空回复)"
        except Exception as e:  # noqa: BLE001 - 结算兜底：确保 inflight 必被清理、future 必被结算
            self._record_result(p.name, ok=False)
            out = f"错误：调用子智能体 '{p.name}' 异常：{type(e).__name__}: {e}"
        with self._lock:
            # 只在"自己仍是该 key 的当前在飞请求"时才清理 inflight。
            # 跟随者超时接管会注册一个新 Future 顶替旧的（见 _follow），旧发起者若
            # 无条件 pop，会把接管者的 inflight 一并删掉——后来的请求于是看不到在飞项，
            # 又各自成为新发起者重复打网络，去重机制在最需要它的挂死场景下正好失效。
            if self._inflight.get(key) is fut:
                self._inflight.pop(key, None)
            if out and not self._is_error(out) and self._cache_size > 0:
                self._cache[key] = out
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)  # 淘汰最久未用
            try:
                fut.set_result(out)  # 锁内结算；try-except 防 Future 已取消等边缘情况
            except Exception:  # noqa: BLE001
                pass
        return out

    def _follow(self, key: tuple, fut: Future, worker: str, prompt: str, system: str | None) -> str:
        """跟随者路径：复用发起者结果，但带在飞超时兜底。

        场景：发起者因网络挂死/超长未返回时，跟随者不应无限阻塞（否则 ask_many/vote
        的外层超时虽能中断调用方，但内部跟随者线程会一直卡在 fut.result() 造成线程泄漏）。
        超时后，只有第一个接管者成为新发起者重新打网络，其余跟随者转等新发起者，避免惊群。
        """
        deadline = time.time() + self._inflight_timeout
        while True:
            remaining = deadline - time.time()
            try:
                return fut.result(timeout=max(0.0, remaining))
            except (TimeoutError, concurrent.futures.TimeoutError):
                with self._lock:
                    if self._inflight.get(key) is fut:
                        # 旧发起者确实卡住：接管，注册新 future 成为新发起者
                        self._inflight.pop(key, None)
                        fut = Future()
                        self._inflight[key] = fut
                        break
                    cur = self._inflight.get(key)
                    if cur is None:
                        # 已被结算（极端竞态：发起者恰在超时瞬间完成并 pop），重新判定入口
                        break
                    fut = cur  # 已被其它跟随者接管，转等新发起者
        # 接管或重判定后：接管者作为发起者执行；竞态落到 None 的重新走 ask 入口（命中缓存）
        if self._inflight.get(key) is fut:
            return self._lead(key, fut, self.profiles[worker], system, prompt)
        return self.ask(worker, prompt, system)

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
        self, futs: dict[Future, str], timeout: float,
        on_done=None, cancel_event=None,
    ) -> dict[str, str | None]:
        """等待一批 {Future: 工人名}，整体限时，返回 {工人: 回答或 None}。

        用 `wait(..., FIRST_COMPLETED)` 边完成边收集，这样可选地支持：
        - `on_done(worker, answer)`：每有一个工人回来就回调一次（ask_many 逐工人
          冒进度事件用）。回调异常被吞，绝不影响派工主流程与最终返回值。
        - `cancel_event`（threading.Event）：置位后停止继续等待——已完成的结果照常
          收集，未完成的立即 cancel + 标记 None 返回（用户「采纳已完成、结束派工」）。
          在途 LLM 请求不追溯撤单（额度已花），由后台线程按 llm.timeout 自行收尾。

        不带这两个参数时（collect/vote/_run_ballot 等）行为与旧的"一把 wait 到底"一致：
        整体到 deadline 或全部完成才返回，超时/未完成的标 None 并 cancel、不 join。
        每轮 wait 用 min(remaining, 0.5) 轮询，保证无工人完成时也能在 ~0.5s 内响应取消。
        """
        fut_to_worker = futs
        deadline = time.time() + timeout
        out: dict[str, str | None] = {}
        pending = set(fut_to_worker)
        while pending:
            if cancel_event is not None and cancel_event.is_set():
                break
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
                except Exception as e:  # noqa: BLE001
                    res = f"失败：{type(e).__name__}: {e}"
                out[w] = res
                if on_done is not None:
                    try:
                        on_done(w, res)
                    except Exception:  # noqa: BLE001 - 进度回调异常不影响派工
                        pass
        for fut in pending:  # 超时或被取消：放弃等待（已启动的后台自行收尾）
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
        on_done=None, cancel_event=None,
    ) -> dict[str, str | None]:
        """并行派工给多个工人并限时收集，调用方拿到后 shutdown(wait=False) 不阻塞。

        on_done / cancel_event 透传给 _wait_gather（只有 ask_many 会用，实现逐工人
        冒进度 + 用户提前结束派工；collect/vote 等不传，保持整批栅栏语义）。
        """
        ex = ThreadPoolExecutor(max_workers=min(len(workers), self.cfg.max_workers))
        try:
            futs = {ex.submit(self.ask, w, prompt, system): w for w in workers}
            return self._wait_gather(futs, timeout, on_done=on_done, cancel_event=cancel_event)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    def collect(
        self, workers: list[str], prompt: str, system: str | None = None,
        timeout: float | None = None,
    ) -> list[tuple[str, str]]:
        """并行收集多个工人对同一任务的成功回答，按入参工人顺序返回（确定性）。

        与 ask_many 的区别：只返回成功回答的 (工人, 回答) 列表，且顺序与
        workers 入参顺序一致，不随完成先后漂移——投票编号、流水线择优
        依赖该确定性（as_completed 完成序会导致同一任务两次运行编号互换）。
        失败/冷却/当日隔离/超时/空回复的工人不进结果。

        这里**不做**冷却/隔离预过滤：过滤交给 ask 内部，才能保证"缓存命中优先于
        冷却判定"（REQUIREMENTS §2.1：缓存命中零成本、不受节点冷却影响）。
        预过滤会让已缓存答案的工人连缓存都查不到就被剔出候选池，与契约相悖。
        """
        valid = [w for w in dict.fromkeys(workers) if w in self.profiles]
        if not valid:
            return []
        got = self._run_parallel(
            valid, prompt, system, self._effective_timeout(len(valid), timeout)
        )
        return [(w, got[w]) for w in valid if got.get(w) and not self._is_error(got[w])]

    def _gather_many(
        self, workers: list[str], prompt: str, system: str | None,
        timeout: float | None, emit, cancel_event,
    ) -> dict:
        """并行派工的收集核心：ask_many（文本）与 ask_many_structured（结构化）共用。

        返回 {valid, got, cancelled, completed, skipped, elapsed, limit}。
        got[worker] 为 None 表示未完成（超时/被取消放弃等待）；否则是 ask 的返回文本
        （成功答案或以「错误：/失败：」开头的说明）。elapsed[worker] 为该工人从派工到
        回来的毫秒数（没回来的没有）。emit / cancel_event 语义同 ask_many（仅工具路径用）。
        """
        valid = [w for w in dict.fromkeys(workers) if w in self.profiles]
        if not valid:
            return {"valid": [], "got": {}, "cancelled": False, "completed": [],
                    "skipped": [], "elapsed": {}, "limit": 0.0}
        limit = self._effective_timeout(len(valid), timeout)
        t0 = time.monotonic()
        completed: list[str] = []
        elapsed: dict[str, int] = {}

        def _on_done(w: str, res: str | None) -> None:
            completed.append(w)
            elapsed[w] = int((time.monotonic() - t0) * 1000)
            if emit is None:
                return
            ok = res is not None and not self._is_error(res)
            try:
                emit({
                    "type": "worker_result", "worker": w,
                    "index": len(completed), "total": len(valid),
                    "elapsed_ms": elapsed[w], "ok": ok, "answer": res or "",
                })
            except Exception:  # noqa: BLE001 - 事件冒泡失败不能影响派工
                pass

        if emit is not None:
            try:
                emit({"type": "worker_dispatch", "workers": list(valid),
                      "total": len(valid), "prompt": prompt})
            except Exception:  # noqa: BLE001
                pass
        got = self._run_parallel(valid, prompt, system, limit, on_done=_on_done, cancel_event=cancel_event)
        cancelled = cancel_event is not None and cancel_event.is_set()
        skipped = [w for w in valid if got.get(w) is None]
        if cancelled and emit is not None:
            try:
                emit({"type": "worker_gather_cancelled",
                      "completed": list(completed), "skipped": skipped})
            except Exception:  # noqa: BLE001
                pass
        return {"valid": valid, "got": got, "cancelled": cancelled, "completed": completed,
                "skipped": skipped, "elapsed": elapsed, "limit": limit}

    def ask_many(
        self, workers: list[str], prompt: str, system: str | None = None,
        timeout: float | None = None, emit=None, cancel_event=None,
    ) -> str:
        """同一子任务并行派给多个工人，收集全部回答（失败也如实带回）。

        输出按入参工人顺序排列（确定性）；timeout 对收集阶段整体限时，缺省按
        参与人数与并发上限自适应（见 _effective_timeout），不用固定值——池大时
        固定值必然截断尾部批次。超时工人以「未完成」如实带回，不无限等待。

        冷却/当日隔离的工人**不在入口预过滤**，交给 ask 内部判定：
        它们返回"跳过"提示并如实带回分节（原因不变），而**已缓存答案的工人
        仍能命中缓存**——预过滤会让缓存命中一起被剔掉，与"缓存命中不受冷却影响"
        的契约冲突。

        emit(event_dict) / cancel_event(threading.Event) 是给 ask_workers 工具用的
        实时反馈通路（不影响 collect/vote/流水线），详见 _gather_many。给主 LLM 的
        始终是完整文本；要程序化消费请用 ask_many_structured。
        """
        r = self._gather_many(workers, prompt, system, timeout, emit, cancel_event)
        valid = r["valid"]
        if not valid:
            return f"错误：没有可用的子智能体。可用：{', '.join(self.profiles) or '无'}"
        got, cancelled, completed, skipped, limit = (
            r["got"], r["cancelled"], r["completed"], r["skipped"], r["limit"])
        blocks: list[str] = []
        if cancelled:
            blocks.append(
                "### 派工已由用户提前结束\n"
                f"用户已采纳已完成的 {len(completed)}/{len(valid)} 份方案；"
                f"以下工人未完成、已放弃等待（其请求可能仍在后台收尾）："
                f"{', '.join(skipped) or '无'}"
            )
        for w in valid:  # 按入参顺序输出，含跳过项
            out = got.get(w)
            if out is None:
                out = (
                    "失败：用户已提前结束派工，未完成（后台线程会自行收尾）"
                    if cancelled else
                    f"失败：超过 {limit:g}s 未完成，已放弃等待（后台线程会自行收尾）"
                )
            else:
                kind = self._skip_kind(out)
                if kind:
                    out = f"错误：工人{kind}，本轮跳过"
            blocks.append(f"### 工人 {w} 的结果\n{out}")
        return "\n\n".join(blocks)

    def ask_many_structured(
        self, workers: list[str], prompt: str, system: str | None = None,
        timeout: float | None = None, emit=None, cancel_event=None,
    ) -> list[dict]:
        """并行派工的结构化结果（外部主程序化消费，替代 Markdown 大块文本）。

        返回按入参顺序的 list[dict]，每项：
          {worker, ok(bool), status, answer(str), elapsed_ms(int|None)}
        status ∈ {"ok","error","冷却中","当日失败隔离","timeout","cancelled"}：
          - ok=True 时 status="ok"、answer 为工人回答；
          - 跳过类取 ask 里已带的分类（冷却中/当日失败隔离），失败为 "error"；
          - 未完成按是否被取消给 "cancelled"/"timeout"（answer 置空）。
        与 ask_many 共用 _gather_many，emit/cancel_event 语义一致。
        """
        r = self._gather_many(workers, prompt, system, timeout, emit, cancel_event)
        got, cancelled, elapsed = r["got"], r["cancelled"], r["elapsed"]
        out: list[dict] = []
        for w in r["valid"]:
            res = got.get(w)
            if res is None:
                out.append({"worker": w, "ok": False,
                            "status": "cancelled" if cancelled else "timeout",
                            "answer": "", "elapsed_ms": elapsed.get(w)})
                continue
            kind = self._skip_kind(res)
            ok = not self._is_error(res)
            out.append({
                "worker": w, "ok": ok,
                "status": kind or ("ok" if ok else "error"),
                "answer": res if ok else "",
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
        for raw in got.values():
            if raw is None:
                continue
            # 错误/跳过类回答不是"投票意见"，必须排除后再抓数字：
            # 冷却提示、HTTP 状态码（如 429）、"超过 300s 未完成" 里都带数字，
            # 用 re.search(r"\d+") 直接抓会把它们算成选票（越界的虽被丢弃，
            # 落在 1..候选数 区间的就会变成一张假票）。
            if self._is_error(raw):
                continue
            m = re.search(r"(?<!\d)\d+(?!\d)", raw)
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


def register_worker_tools(registry: ToolRegistry, pool: WorkerPool) -> None:
    """把派工能力注册为主智能体的工具。"""
    registry.register(FunctionTool(
        name="ask_worker",
        description=(
            "派一个子任务给指定的子智能体（纯文本执行，它没有本地工具）。"
            "适合把独立子任务分出去并行干、或对某个问题要第二意见。"
            "返回该工人的回答文本。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "worker": {"type": "string", "description": "子智能体名"},
                "prompt": {
                    "type": "string",
                    "description": "完整、自包含的子任务描述。工人看不到对话历史，所有必要上下文、格式要求都要写全。",
                },
                "system": {"type": "string", "description": "可选，给工人的角色设定"},
            },
            "required": ["worker", "prompt"],
        },
        func=lambda a: pool.ask(str(a.get("worker", "")), str(a.get("prompt", "")), a.get("system")),
    ))
    registry.register(FunctionTool(
        name="ask_workers",
        description=(
            "把同一个子任务并行派给多个子智能体，收集全部回答（含失败说明）。"
            "适合多方案对比、交叉验证、投票取共识。结果按工人分节返回。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "workers": {"type": "array", "items": {"type": "string"}, "description": "子智能体名列表"},
                "prompt": {"type": "string", "description": "完整、自包含的子任务描述"},
                "system": {"type": "string", "description": "可选，给工人的角色设定"},
            },
            "required": ["workers", "prompt"],
        },
        func=_make_ask_workers_func(pool),
    ))
    registry.register(FunctionTool(
        name="ask_vote",
        description=(
            "两步投票取共识：先让多名子智能体各自给出方案，再让全体对方案编号投票，"
            "超阈值（默认 collaboration.vote_threshold）即宣布共识并给出胜出方案全文。"
            "适合方案选型、结论判断等需要多数共识的关键决策。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "需要共识的具体问题，自包含、写清评判标准"},
                "workers": {
                    "type": "array", "items": {"type": "string"},
                    "description": "参与的子智能体名列表，缺省用池内全部可用工人",
                },
                "threshold": {
                    "type": "number", "description": "达成共识的票数占比阈值（0-1），缺省用配置项",
                },
            },
            "required": ["prompt"],
        },
        func=lambda a: pool.vote(
            str(a.get("prompt", "")),
            [str(w) for w in (a.get("workers") or [])] or None,
            a.get("threshold"),
        ),
    ))


def _make_ask_workers_func(pool: WorkerPool):
    """给 ask_workers 工具造一个「双参 (args, ctx)」执行函数。

    双参签名是刻意的：FunctionTool 按签名自动决定是否注入 ctx（见 tools/base.py）。
    ctx 是 Agent 每次工具调用现造的小对象，带 `.emit(event)` 与 `.cancel_event`
    （threading.Event 或 None）：
      - emit 让派工逐工人冒实时进度事件（快工人先亮，用户不必等最慢的）；
      - cancel_event 被 Agent.cancel_current_tool() 置位后，ask_many 停止继续等待，
        把已完成的部分作为工具结果交给主 LLM 继续（「采纳已完成、结束派工」）。
    ctx 为 None（bridge call_tool / 单测 / 无事件通道）时退化成纯栅栏调用，与旧行为一致。
    """

    def _func(args: dict, ctx=None) -> str:
        workers = [str(w) for w in (args.get("workers") or [])]
        prompt = str(args.get("prompt", ""))
        system = args.get("system")
        emit = getattr(ctx, "emit", None) if ctx is not None else None
        cancel_event = getattr(ctx, "cancel_event", None) if ctx is not None else None
        return pool.ask_many(workers, prompt, system, emit=emit, cancel_event=cancel_event)

    return _func
