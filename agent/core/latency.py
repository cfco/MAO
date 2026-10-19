"""模型延迟档案：记录每个工人的最近往返延迟（取中位数），供 pick() 按延迟选路。

为什么单独一层（与 health.py 并列）：
- health 管"能不能用"（当日隔离 / 连续失败下线）；本模块管"快不快用"——同一批可用
  工人里，延迟高的先不派，把有限派工机会让给低延迟节点。免费中转站常有"能用但很慢"
  的模型，占着派工位拖慢整批。
- 延迟事实源只有这一份：LatencyProber 每 latency_probe_interval 秒对所有节点（含当前
  不可用节点）主动探测刷新，真实调用成功时也顺带回填——两者共写本档案（"一套"）。
- 选路规则（问题5，一套公式）：可用工人按延迟升序排，只留延迟最低的
  latency_keep_ratio（默认 0.6）——弃掉最高延迟的 40%；小池上同一公式落在"用最低的
  2~3 个"。证据不足（已知样本 < latency_min_samples）时不动刀。见 orchestrator.pick()。

进程内共享（get_latency）：与 health 同理，同一档案路径全进程唯一实例，避免多池各持
账本互相覆写（lost update）。落盘只记延迟数值，不含任何密钥。
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from pathlib import Path

from .ledger import AtomicJsonLedger, shared_by_path
from .llm import LLMError

_MAX_SAMPLES = 5  # 每工人保留最近 N 次延迟样本（中位数抗单次抖动）
_VIEW_KEEP = 500  # 档案最多保留的工人条数（防异常膨胀）
# 探测用的极短提示：只要一次最小往返，测的是"节点响应快不快"、不关心回答内容。
# 空回复视为无效往返（节点不可信，不计入延迟样本，见 LatencyMixin._probe_one）。
_PROBE_PROMPT = "ping"
# 探测等同站串行门的最长等待（秒）：门被真实派工占着（可达 llm.timeout 量级）时，
# 探测**不陪等**——拿不到门就本轮跳过该工人。理由：探测器是后台顺带动作，为一两个
# 样本把整轮 join 挂几分钟，既拖内存里在跑的派工，也没必要。
_PROBE_GATE_WAIT = 2.0


def _median(samples: list) -> float | None:
    """样本中位数；空 / 全非法返回 None（不抛异常）。"""
    nums: list[float] = []
    for s in samples:
        try:
            nums.append(float(s))
        except (TypeError, ValueError):
            continue
    if not nums:
        return None
    return round(statistics.median(nums), 1)


def _trim(data: dict[str, dict]) -> dict[str, dict]:
    """只保留最近更新的 _VIEW_KEEP 个工人条目（按 updated_at），防档案无限膨胀。"""
    if len(data) <= _VIEW_KEEP:
        return data
    items = sorted(data.items(), key=lambda kv: kv[1].get("updated_at", 0.0))
    return dict(items[-_VIEW_KEEP:])


class LatencyStore(AtomicJsonLedger):
    """模型延迟账本：最近 N 次往返延迟样本 → 中位数。热路径只读内存。"""

    def __init__(self, store_path: Path):
        # _lock / _write_lock / store_path 与落盘口径由基类提供（见 ledger.py）
        super().__init__(store_path)
        self._data: dict[str, dict] | None = None  # 懒加载：首次使用时才读盘
        self._load_lock = threading.Lock()  # 只护"首次读盘"，与 _lock 分开（见 _ensure_loaded）
        self._snapshot_cache: dict[str, float] | None = None  # 中位数结果缓存（见 snapshot）

    # ---------- 查询 ----------

    def get_ms(self, name: str) -> float | None:
        """该工人当前延迟（最近样本中位数，毫秒）；无样本返回 None。只查内存。"""
        self._ensure_loaded()  # 读盘放在 _lock 之外：载入不在记账锁内碰磁盘
        with self._lock:
            entry = self._data.get(name) or {}
            return _median(entry.get("samples") or [])

    def snapshot(self) -> dict[str, float]:
        """{worker: 延迟中位数ms}（只含有样本的工人），供 pick 排序与诊断。

        中位数结果带缓存：pick 每次派工都会调它，而样本只有在 record/record_many 时才变
        （那时缓存随之作废），没必要每次把全池样本重算一遍中位数。返回一份拷贝，
        调用方可随意读。
        """
        if self._snapshot_cache is not None:
            return dict(self._snapshot_cache)
        self._ensure_loaded()  # 读盘放在 _lock 之外
        with self._lock:
            out: dict[str, float] = {}
            for k, v in self._data.items():
                ms = _median(v.get("samples") or [])
                if ms is not None:
                    out[k] = ms
            self._snapshot_cache = out
        return dict(out)

    # ---------- 记录 ----------

    def record(self, name: str, ms: float) -> None:
        """记一次往返延迟（毫秒）：追加样本、保留最近 _MAX_SAMPLES 条，再落盘。

        落盘放在锁外（自带 _write_lock）：get_ms/snapshot 是 pick 热路径，绝不能被磁盘
        写拖住。写入失败只损失持久化，内存账本照常工作。
        """
        self.record_many([(name, ms)])

    def record_many(self, samples: list[tuple[str, float]]) -> None:
        """批量记一批样本，**只落盘一次**（探测轮用：N 个工人各写一遍整份档案就是 N 次 IO）。

        与 record 同一口径：单条非法值（非数字 / 负数）跳过该条、其余照记；一条都没记成
        就不落盘。每工人保留最近 _MAX_SAMPLES 条；落盘在锁外（见 ledger.write）。
        """
        self._ensure_loaded()  # 读盘放在 _lock 之外
        added = False
        with self._lock:
            for name, ms in samples:
                try:
                    value = float(ms)
                except (TypeError, ValueError):
                    continue
                if value < 0:
                    continue
                entry = self._data.setdefault(name, {})
                arr = list(entry.get("samples") or [])
                arr.append(round(value, 1))
                entry["samples"] = arr[-_MAX_SAMPLES:]
                entry["updated_at"] = time.time()
                added = True
            if added:
                self._data = _trim(self._data)
                self._snapshot_cache = None  # 样本变了：作废中位数缓存
        if added:
            self.write()  # 快照+落盘都在 _write_lock 内串行，见 ledger.AtomicJsonLedger.write

    # ---------- 落盘 ----------

    def _ensure_loaded(self) -> None:
        """保证内存账本已载入盘上数据。**必须在 _lock 之外调用**（载入要读盘）。

        持 _lock 读盘会把慢盘串进 pick 热路径（ledger.AtomicJsonLedger 的"锁内绝不碰磁盘"
        口径）。这里不碰 _lock，只用 _load_lock 防并发首查各读一遍盘；已载入的快速路径
        连 _load_lock 都不取。写盘路径由基类 write() 保证先载入再进 _lock。
        """
        if self._data is not None:
            return
        with self._load_lock:
            if self._data is not None:
                return
            data: dict[str, dict] = {}
            try:
                raw = json.loads(self.store_path.read_text(encoding="utf-8"))
                models = raw.get("models") if isinstance(raw, dict) else None
                if isinstance(models, dict):
                    data = {k: v for k, v in models.items() if isinstance(v, dict)}
            except (OSError, ValueError):
                data = {}  # 缺失/损坏都从空白账本重新开始，不拖垮派工
            self._data = data

    def _snapshot_locked(self) -> str:  # 调用方持锁
        """内存账本 → 待写文本。锁内只读内存，不碰磁盘。"""
        payload = {"version": 1, "models": self._data}
        return json.dumps(payload, ensure_ascii=False, indent=2)


class LatencyProber:
    """延迟探测器：每 interval 秒对**全部工人（含当前不可用节点）**做一轮轻量往返测量。

    为什么懒式定时（不自建常驻定时线程）：MAO 多为短命外壳（CLI 一次调用即退出），
    常驻定时线程会在无人派工时空转、白烧免费额度。改为"派工 / 预检入口顺带检查是否
    到点"——使用期内仍是每 interval 秒刷新一轮（"每 10 分钟自动探测一次所有节点"），
    空载不探测。首轮立即到期（_last_start=0），冷启动也能尽快拿到基线。

    探测跑在守护线程里，绝不阻塞派工主流程；**同一时刻只允许一轮在飞**：自动轮由
    _claim 占名额（并发的派工不会各起一轮），同步强制刷新由 _round_lock 排队
    （见 probe_all）。探测只写延迟档案，**不碰**冷却与当日隔离：延迟是"快不快"的事实，
    改不了"能不能用"的判定（那是 health 的职责），否则定时器会变成自己给自己制造隔离的噪声源。
    """

    def __init__(
        self, store: LatencyStore, interval: float, probe_one, concurrency: int = 5, clock=time.time
    ):
        """store       延迟档案（探测结果写入此处）
        interval    探测周期（秒）
        probe_one   (工人名) -> 毫秒或 None；由调用方注入真实测量逻辑（可测）
        concurrency 单轮并发上限（防一轮探测同时打出几十个请求）
        clock       时间源（测试可注入，免等真实时间流逝）
        """
        self._store = store
        self._interval = max(1.0, float(interval))
        self._probe_one = probe_one
        self._concurrency = max(1, int(concurrency))
        self._clock = clock
        self._lock = threading.Lock()
        # 整轮串行锁：自动轮与同步强制刷新都要过这把锁，保证绝不会有两轮同时在打网络。
        # 嵌套方向恒为 _round_lock → _lock，反向不存在（_lock 临界区内不碰轮锁）。
        self._round_lock = threading.Lock()
        self._last_start = 0.0  # 0 = 从未探测 ⇒ 首次检查即到期
        self._last_finish = 0.0
        self._running = False
        self._rounds = 0

    # ---------- 调度 ----------

    def due(self) -> bool:
        """当前是否该起一轮新探测（到期且无在飞）。"""
        with self._lock:
            return not self._running and (self._clock() - self._last_start) >= self._interval

    def maybe_start(self, names) -> bool:
        """到点且无在飞 ⇒ 起守护线程跑一轮；返回是否真的起了（调用方不等待）。"""
        if not self._claim():
            return False
        try:
            threading.Thread(target=self.probe_all, args=(list(names),), daemon=True).start()
        except (RuntimeError, OSError):
            # 线程起不来（系统线程资源耗尽等）必须把名额还回去，否则 _running 永久为
            # True、自动探测从此静默停摆且没有自愈机会。回滚后下个检查点照旧到期重试。
            self._release_claim()
            return False
        return True

    def _claim(self) -> bool:
        """占住本轮名额（原子）：到点且无在飞才允许开跑，避免并发的派工各起一轮。"""
        with self._lock:
            now = self._clock()
            if self._running or (now - self._last_start) < self._interval:
                return False
            self._running = True
            self._last_start = now
            return True

    def _release_claim(self) -> None:
        """回滚一次没跑成的占位：周期起算点退回上一轮结束时间（从未跑过则为 0 ⇒ 立即再到期）。"""
        with self._lock:
            self._running = False
            self._last_start = self._last_finish

    # ---------- 探测 ----------

    def probe_all(self, names) -> dict:
        """同步跑完一轮探测，返回 {工人: 毫秒 或 None}（外部主强制刷新 / 测试用）。

        **绝不与另一轮同时在飞**：自动轮由 _claim 把关，但这里是同步入口、不经 _claim，
        撞上后台轮就会两轮同时打网络——白烧额度、抢同一批站门，而且样本测的是站端排队
        （档案是裁剪选路的依据，排名会被系统性带偏）。_round_lock 让后到的一轮排队；
        _running 由真正在跑的那轮复位，不再出现"先结束的清标志、另一轮还在飞"。
        """
        with self._round_lock:
            with self._lock:
                # 强制刷新同样构成一轮：周期起算点就此刷新，避免紧接着又叠一轮自动探测
                self._running = True
                self._last_start = self._clock()
            try:
                return self._run_round(names)
            finally:
                with self._lock:
                    self._running = False
                    self._last_finish = self._clock()
                    self._rounds += 1

    def _run_round(self, names) -> dict:
        """一轮探测的实际执行体（调用方已持 _round_lock，保证单轮在飞）。

        每个工人独立探测，单个失败/超时不牵连其它工人；成功样本写进档案。
        探测失败**不**记 health / 冷却（见类文档）。
        """
        out: dict[str, float | None] = {}
        lock = threading.Lock()
        sem = threading.Semaphore(self._concurrency)
        waiters: list[threading.Thread] = []
        samples: list[tuple[str, float]] = []

        def _one(nm: str) -> None:
            with sem:
                try:
                    ms = self._probe_one(nm)
                except Exception:  # noqa: BLE001 - 单个工人探测异常不影响整轮
                    ms = None
            with lock:
                out[nm] = ms
                if ms is not None:
                    samples.append((nm, ms))

        for nm in dict.fromkeys(names):  # 去重且保序
            t = threading.Thread(target=_one, args=(nm,), daemon=True)
            waiters.append(t)
            t.start()
        for t in waiters:
            t.join()
        # 整轮只落盘一次：旧实现每个工人各自 record() 各写一遍整份 JSON（N 工人 = N 次 IO）。
        if samples:
            try:
                self._store.record_many(samples)
            except Exception:  # noqa: BLE001 - 记账失败只丢持久化
                pass
        return out

    def stats(self) -> dict:
        """探测调度状态（诊断用，不含密钥）。"""
        with self._lock:
            return {
                "interval_s": self._interval,
                "rounds": self._rounds,
                "running": self._running,
                "last_start": self._last_start,
                "last_finish": self._last_finish,
            }


class LatencyMixin:
    """延迟维度能力（延迟选路 + 定时探测），混入 WorkerPool 使用。

    与 ParallelMixin / VotingMixin 同模式：本类不单独实例化，方法通过 self 访问
    WorkerPool 的 profiles / _client_for 等属性；独立成 Mixin 是为了让 orchestrator.py
    不越「单文件 ≤500 行」约定（延迟选路这段与"派工/健康/pick 轮转"职责不同）。
    """

    def _init_latency(self, cfg, latency: LatencyStore | None = None) -> None:
        """建延迟档案与探测器（由 WorkerPool.__init__ 调用一次）。

        档案默认走 get_latency()：同一路径全进程共享一份账本（理由同 get_health——
        多池各持账本会互相覆盖写盘、且看不到彼此的新鲜样本）。测试可注入独立实例。
        """
        self.latency = latency if latency is not None else get_latency(cfg.model_latency_path)
        self._latency_keep_ratio = cfg.latency_keep_ratio
        self._latency_min_samples = cfg.latency_min_samples
        # 探测请求输出上限：不带时样本混入"模型爱写多长"，话痨模型会被系统性记慢。<=0 不带。
        self._latency_probe_max_tokens = cfg.latency_probe_max_tokens
        # 已探明"不接受 max_tokens 参数"的站（按站键）：免费网关行为不一，遇到参数类
        # 4xx 就按站降级为不带该参数，而不是让整轮探测连续失败、丢光样本。
        self._probe_no_max_tokens: set[tuple[str, str]] = set()
        # 探测专用短超时（秒）：探测客户端不沿用真实派工的 llm.timeout（120s）+ 重试，
        # 否则坏节点会把一次测速拖成分钟级、更会挂住同步 probe 整轮（见 _probe_client_for）。
        self._latency_probe_timeout = cfg.latency_probe_timeout
        # 探测器常驻（外部主的 probe_latency_now 随时可用），但**自动**触发由 latency_probe
        # 把关：每 latency_probe_interval 秒对全部工人（含当前不可用节点）打一次极短往返
        # （用户口径：每 10 分钟自动探测一次所有节点）。关掉它 = 不自动探测，
        # 离线/测试场景必须关（否则首个派工就会起后台线程真打网络）。
        self._latency_auto_probe = cfg.latency_probe
        self._prober = LatencyProber(self.latency, cfg.latency_probe_interval, self._probe_one)

    # ---------- 选路 ----------

    def _latency_sorted(self, names: list[str]) -> list[str]:
        """按延迟升序**稳定**排序（未测过的用已知样本中位数作中性先验）。

        未测过的工人用中性先验而不是"最慢"或"最快"：既不该因为没被测过就被优先放行
        （挤掉已证实快的），也不该被当成最慢的一刀切掉。稳定排序 ⇒ 同延迟的工人保持
        池内相对顺序；无任何样本时逐字返回原列表（旧行为不变）。
        """
        lat = self.latency.snapshot()
        known = [w for w in names if w in lat]
        if not known:
            return list(names)
        prior = float(statistics.median([lat[w] for w in known]))
        return sorted(names, key=lambda w: lat.get(w, prior))

    def _latency_window(self, live: list[str]) -> list[str]:
        """按延迟裁剪候选并排序：只留延迟最低的一批（默认 60%），优先派快节点。

        规则（一套公式，大池小池通用）：drop = floor(n × (1 - ratio))，保留 n - drop 个、
        至少留 1 个。ratio=0.6 时——
          n>5：只用延迟最低的 ~60%（即"不使用高延迟的 40% 模型"）；
          n=5→3、4→3、3→2、2→2：恰好是"可用低于 5 个就用延迟最低的 2~3 个"。
        证据不足（已知延迟的候选少于 latency_min_samples）时原样返回：冷启动、档案为空
        或节点全不可达时绝不能凭空把工人踢出派工，等探测补上样本再裁。
        """
        if len(live) <= 1:
            return live
        lat = self.latency.snapshot()
        known = [w for w in live if w in lat]
        if len(known) < min(self._latency_min_samples, len(live)):
            return live
        ordered = self._latency_sorted(live)
        drop = int(len(ordered) * (1.0 - self._latency_keep_ratio))
        return ordered[: max(1, len(ordered) - drop)]

    # ---------- 探测与记账 ----------

    def _probe_one(self, name: str) -> float | None:
        """对单个工人做一次极短往返，返回毫秒（失败/空回复/让路返回 None）。

        只测速、不记账：探测失败**不**进冷却、不当日隔离——那是真实派工的职责。
        否则"定时探测器"会变成自己给自己制造隔离与冷却的噪声源：节点慢一点被探成失败、
        进冷却被跳过，反而更快被摘掉。
        用探测**专用**短超时、不重试的客户端（_probe_client_for，cfg.latency_probe_timeout），
        不沿用真实派工的 llm.timeout(120s)+重试——坏节点最多拖本值秒、且只发一次请求。

        同站串行（collaboration.max_per_station）对探测同样生效，而且**更要生效**：档案是
        按延迟裁剪选路的依据，若探测请求和真实派工挤在同一条站上并发，测出来的"慢模型"
        其实是站端排队，排名会被系统性带偏。故进门前先拿站门，拿不到（>2s 还被占着）就
        本轮跳过这个工人，绝不陪真实请求等几分钟。
        """
        p = self.profiles.get(name)
        if p is None:
            return None
        station = p.station_key
        gate = self._station_gate(station)
        if not gate.acquire(timeout=_PROBE_GATE_WAIT):
            return None  # 该站正在跑真实请求 ⇒ 本轮让路，不污染样本也不挂住探测轮
        messages = [{"role": "user", "content": _PROBE_PROMPT}]
        client = self._probe_client_for(p)
        try:
            t0 = time.monotonic()
            try:
                resp = client.chat_once(messages, max_tokens=self._probe_max_tokens_for(station))
            except TypeError:
                # 自定义/替身客户端可能只接受 messages（max_tokens 是探测的附加项，
                # 不能因为对方签名旧就让整轮测速全废）；退回标准调用即可。
                resp = client.chat_once(messages)
            except LLMError as e:
                # 该站可能不认 max_tokens（免费网关常回 400/参数错误）：按站降级后重测一次。
                # 只对"非可重试"的失败降级——429/超时/5xx 与参数无关，重测只是白打一次。
                if not self._degrade_probe_params(station, e):
                    return None
                t0 = time.monotonic()  # 降级重测的耗时才是这一站的真实往返
                resp = client.chat_once(messages)
            if not str(resp.get("content") or "").strip():
                return None  # 空回复不算有效往返，避免污染样本
            return (time.monotonic() - t0) * 1000.0
        except Exception:  # noqa: BLE001 - 单个工人探测异常不影响整轮
            return None
        finally:
            gate.release()

    def _probe_max_tokens_for(self, station: tuple[str, str]) -> int | None:
        """本站在探测请求里要不要带 max_tokens（0 / 已探明不支持 ⇒ None=不带）。"""
        if self._latency_probe_max_tokens <= 0 or station in self._probe_no_max_tokens:
            return None
        return self._latency_probe_max_tokens

    def _degrade_probe_params(self, station: tuple[str, str], err: LLMError) -> bool:
        """遇到"该站不认 max_tokens"类失败时按站降级；返回是否值得不带参数重测一次。

        判据只用"是否可重试"：参数错 4xx 在 llm._once 里归非重试，429/超时/5xx 归可重试
        ——后者与参数无关，降级重测只是白烧额度。宁可误判成"不支持"（该站之后都不带，
        样本仍可用），也别让不兼容的网关把整轮探测打成失败、把延迟档案清空。
        """
        if self._latency_probe_max_tokens <= 0 or station in self._probe_no_max_tokens:
            return False
        if err.retryable:
            return False
        self._probe_no_max_tokens.add(station)
        return True

    def _record_latency(self, name: str, ms: float) -> None:
        """把一次真实成功派工的往返耗时写进延迟档案（失败只丢持久化，绝不影响派工）。"""
        try:
            self.latency.record(name, ms)
        except Exception:  # noqa: BLE001 - 延迟记账绝不拖垮派工主流程
            pass

    def _touch_latency(self) -> None:
        """派工入口的顺带动作：探测到期就起一轮后台探测（不阻塞调用方、不抛异常）。

        为什么挂在派工入口而不是自建常驻定时线程：MAO 多为短命外壳（CLI 一次调用即退出），
        常驻线程会在无人派工时空转烧免费额度；挂在入口上则"使用期内每 interval 秒刷新
        一轮"，空载不探测。latency_probe=false（或 MAO_LATENCY_PROBE=0）时整段跳过。
        """
        if not self._latency_auto_probe:
            return
        try:
            self._prober.maybe_start(list(self.profiles))
        except Exception:  # noqa: BLE001 - 探测调度绝不拖累派工
            pass

    def probe_latency_now(self) -> dict:
        """**同步**强制跑一轮全节点探测并返回 {工人: 毫秒 或 None}。

        给外部主/诊断用：新增了节点、或刚怀疑"某个模型变慢了"时立即刷新一遍，
        不必等自动探测的 10 分钟周期。与自动探测同一份档案、同一套测量口径，
        且同一时刻只跑一轮（撞上在飞的自动轮就在 _round_lock 上排队，见 probe_all）。
        """
        return self._prober.probe_all(list(self.profiles))

    def latency_status(self) -> dict:
        """延迟维度诊断快照：{model_latency: {工人: 毫秒}, prober: {...},
        auto_probe: 是否自动探测}（不含密钥）。"""
        return {
            "model_latency": self.latency.snapshot(),
            "prober": self._prober.stats(),
            "auto_probe": self._latency_auto_probe,
        }


# ---------- 进程内共享（同一档案路径复用同一实例） ----------

_shared_lock = threading.Lock()
_shared_instances: dict[Path, LatencyStore] = {}


def get_latency(store_path: Path) -> LatencyStore:
    """按档案路径取进程内共享的 LatencyStore（WorkerPool 默认构造入口）。

    与 get_health 同理：同进程多 WorkerPool 时共享同一账本，避免各自覆写丢更新。
    直接 LatencyStore(...) 仍返回独立实例，保留"新实例=模拟新进程重读盘"的可测性。
    """
    key = Path(store_path)
    return shared_by_path(_shared_instances, _shared_lock, key, lambda: LatencyStore(key))
