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
# 批量派工的整体限时不再是固定常量：按「批次 × 单请求超时」自适应推导，
# 见 WorkerPool._effective_timeout —— 固定值在「池大 + 并发低」时必然截断尾部批次。
# 单请求超时直接取 llm.timeout，与主智能体、工人客户端同源。


class WorkerPool:
    """子智能体池：纯文本执行器（不挂本地工具，兼容性最好、最安全）。

    免费节点友好：单节点抖动/限流/不可用不拖垮整体——连续失败进入冷却期，
    冷却中跳过该节点（ask 直接返回冷却提示），其它节点照常工作。
    健康档案（ModelHealth）叠加在冷却之上：某模型当天出现终态失败，当天不再派工给它。
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

        失败同时记入持久健康档案（ModelHealth）：当天不再向该模型派工。
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

        免费节点友好：节点在冷却期、或当天已出现终态失败（当日隔离）时直接快速
        跳过并提示，不傻等。
        说明：历史上这里还有 LRU 缓存与「并发同 key 去重（在飞 Future 复用 + 超时
        接管）」；外部主按需发话、prompt 几乎不逐字重复，两者命中率极低、且是全套
        里最复杂的并发代码，已随「单一 bridge 路线」简化一并移除——每次如实打网络。
        """
        p = self.profiles.get(worker)
        if not p:
            return f"错误：子智能体 '{worker}' 不存在。可用：{', '.join(self.profiles) or '无'}"
        reason = self._skip_reason(worker)
        if reason == "冷却中":
            return f"错误：子智能体 '{worker}' 近期连续失败，正在冷却中，建议稍后再试或换其它工人"
        if reason == "当日失败隔离":
            return (
                f"错误：子智能体 '{worker}' 今日调用已失败，当天隔离不再派工，"
                "请换其它工人或明天再试"
            )
        return self._call(p, system, prompt)

    def _call(self, p: AgentProfile, system: str | None, prompt: str) -> str:
        """真正打一次网络：取（或复用）该工人的 LLMClient → chat → 记账 → 归一返回。

        任何失败（LLMError / 其它异常）都计入节点健康冷却并返回「错误：…」文本，
        不抛异常；ask_many/collect/vote 依赖这个「永远返回字符串」的契约。
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
                self._record_result(p.name, ok=False)
                return f"错误：调用子智能体 '{p.name}' 失败 [{e.error_type}]：{e}"
            except Exception as e:  # noqa: BLE001 - 兜底：其它异常也计入失败
                self._record_result(p.name, ok=False)
                return f"错误：调用子智能体 '{p.name}' 失败：{type(e).__name__}: {e}"
            self._record_result(p.name, ok=True)
            return resp.get("content", "") or "(空回复)"
        except Exception as e:  # noqa: BLE001 - 连取 client 都失败，也不让派工崩
            self._record_result(p.name, ok=False)
            return f"错误：调用子智能体 '{p.name}' 异常：{type(e).__name__}: {e}"

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
    ) -> dict[str, str | None]:
        """等待一批 {Future: 工人名}，整体限时，返回 {工人: 回答或 None}。

        用 `wait(..., FIRST_COMPLETED)` 边完成边收集；`on_done(worker, answer)` 每有
        一个工人回来就回调一次（ask_many 用它记每个工人的耗时）。回调异常被吞。
        到 deadline 时未完成的工人 cancel + 标记 None（已启动的按 LLM 超时后台自行收尾），
        调用方不 join。
        """
        fut_to_worker = futs
        deadline = time.time() + timeout
        out: dict[str, str | None] = {}
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
                except Exception as e:  # noqa: BLE001
                    res = f"失败：{type(e).__name__}: {e}"
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
    ) -> dict[str, str | None]:
        """并行派工给多个工人并限时收集，调用方拿到后 shutdown(wait=False) 不阻塞。

        on_done 透传给 _wait_gather（ask_many 用它记每个工人耗时；collect/vote 不传）。
        """
        ex = ThreadPoolExecutor(max_workers=min(len(workers), self.cfg.max_workers))
        try:
            futs = {ex.submit(self.ask, w, prompt, system): w for w in workers}
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

        这里**不做**冷却/隔离预过滤：判定交给 ask 内部，返回"跳过"提示文本，
        由下面的 `_is_error` 统一过滤掉——保持"过滤只在 ask 一处"的单一职责。
        """
        valid = [w for w in dict.fromkeys(workers) if w in self.profiles]
        if not valid:
            return []
        got = self._run_parallel(
            valid, prompt, system, self._effective_timeout(len(valid), timeout)
        )
        return [(w, got[w]) for w in valid if got.get(w) and not self._is_error(got[w])]

    def _gather_many(
        self, workers: list[str], prompt: str, system: str | None, timeout: float | None,
    ) -> dict:
        """并行派工的收集核心：ask_many（文本）与 ask_many_structured（结构化）共用。

        返回 {valid, got, elapsed, limit}。got[worker] 为 None 表示超时未完成；
        否则是 ask 的返回文本。elapsed[worker] 为该工人从派工到回来的毫秒数。
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
        冷却/当日隔离工人不预过滤，交给 ask 内部判定并如实带回跳过说明。
        """
        r = self._gather_many(workers, prompt, system, timeout)
        valid = r["valid"]
        if not valid:
            return f"错误：没有可用的子智能体。可用：{', '.join(self.profiles) or '无'}"
        got, limit = r["got"], r["limit"]
        blocks: list[str] = []
        for w in valid:  # 按入参顺序输出，含跳过项
            out = got.get(w)
            if out is None:
                out = f"失败：超过 {limit:g}s 未完成，已放弃等待（后台线程会自行收尾）"
            else:
                kind = self._skip_kind(out)
                if kind:
                    out = f"错误：工人{kind}，本轮跳过"
            blocks.append(f"### 工人 {w} 的结果\n{out}")
        return "\n\n".join(blocks)

    def ask_many_structured(
        self, workers: list[str], prompt: str, system: str | None = None,
        timeout: float | None = None,
    ) -> list[dict]:
        """并行派工的结构化结果（外部主程序化消费，替代 Markdown 大块文本）。

        返回按入参顺序的 list[dict]，每项 {worker, ok(bool), status, answer, elapsed_ms}。
        status ∈ {"ok","error","冷却中","当日失败隔离","timeout"}：跳过类取 ask 里已带的
        分类，失败为 "error"（answer 置空），超时未完成为 "timeout"。
        """
        r = self._gather_many(workers, prompt, system, timeout)
        got, elapsed = r["got"], r["elapsed"]
        out: list[dict] = []
        for w in r["valid"]:
            res = got.get(w)
            if res is None:
                out.append({"worker": w, "ok": False, "status": "timeout",
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

