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
import os
import statistics
import threading
import time
from pathlib import Path

_MAX_SAMPLES = 5      # 每工人保留最近 N 次延迟样本（中位数抗单次抖动）
_VIEW_KEEP = 500      # 档案最多保留的工人条数（防异常膨胀）
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


class LatencyStore:
    """模型延迟账本：最近 N 次往返延迟样本 → 中位数。热路径只读内存。"""

    def __init__(self, store_path: Path):
        self.store_path = Path(store_path)
        self._lock = threading.Lock()
        # 写盘专用锁：**取快照 + 写盘**整段串行（见 _write_payload）。写盘在 _lock 之外
        # （get_ms/snapshot 是热路径）；若快照取在 _write_lock 之外，写盘顺序可能与快照
        # 顺序相反，旧快照会覆盖新快照丢样本。嵌套方向恒为 write_lock→lock。
        self._write_lock = threading.Lock()
        self._data: dict[str, dict] | None = None  # 懒加载：首次使用时才读盘

    # ---------- 查询 ----------

    def get_ms(self, name: str) -> float | None:
        """该工人当前延迟（最近样本中位数，毫秒）；无样本返回 None。只查内存。"""
        with self._lock:
            self._ensure_loaded()
            entry = self._data.get(name) or {}
            return _median(entry.get("samples") or [])

    def snapshot(self) -> dict[str, float]:
        """{worker: 延迟中位数ms}（只含有样本的工人），供 pick 排序与诊断。"""
        with self._lock:
            self._ensure_loaded()
            out: dict[str, float] = {}
            for k, v in self._data.items():
                ms = _median(v.get("samples") or [])
                if ms is not None:
                    out[k] = ms
            return out

    # ---------- 记录 ----------

    def record(self, name: str, ms: float) -> None:
        """记一次往返延迟（毫秒）：追加样本、保留最近 _MAX_SAMPLES 条，再落盘。

        落盘放在锁外（自带 _write_lock）：get_ms/snapshot 是 pick 热路径，绝不能被磁盘
        写拖住。写入失败只损失持久化，内存账本照常工作。
        """
        try:
            value = float(ms)
        except (TypeError, ValueError):
            return
        if value < 0:
            return
        with self._lock:
            self._ensure_loaded()
            entry = self._data.setdefault(name, {})
            samples = list(entry.get("samples") or [])
            samples.append(round(value, 1))
            entry["samples"] = samples[-_MAX_SAMPLES:]
            entry["updated_at"] = time.time()
            self._data = _trim(self._data)
        self._write_payload()  # 快照+落盘都在 _write_lock 内串行，见 _write_payload

    # ---------- 落盘 ----------

    def _ensure_loaded(self) -> None:  # 调用方持锁
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

    def _write_payload(self) -> None:
        """取账本快照并原子写入档案（**_lock 之外调用**）：临时文件 + os.replace，失败只丢持久化。

        快照在 _write_lock **之内**取（与 health._write_payload 同一口径）：分开取的话两个
        线程的写盘顺序可能与快照顺序相反，旧快照覆盖新快照丢样本。落盘 IO 仍留在 _lock 之外，
        get_ms/snapshot 不被慢盘拖住。
        """
        tmp: Path | None = None
        with self._write_lock:
            with self._lock:
                text = self._snapshot_locked()
            try:
                self.store_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.store_path.with_name(
                    f"{self.store_path.name}.{os.getpid()}-{threading.get_ident()}.tmp"
                )
                tmp.write_text(text, encoding="utf-8")
                os.replace(tmp, self.store_path)
            except OSError:
                if tmp is not None:
                    try:
                        tmp.unlink(missing_ok=True)
                    except OSError:
                        pass


class LatencyProber:
    """延迟探测器：每 interval 秒对**全部工人（含当前不可用节点）**做一轮轻量往返测量。

    为什么懒式定时（不自建常驻定时线程）：MAO 多为短命外壳（CLI 一次调用即退出），
    常驻定时线程会在无人派工时空转、白烧免费额度。改为"派工 / 预检入口顺带检查是否
    到点"——使用期内仍是每 interval 秒刷新一轮（"每 10 分钟自动探测一次所有节点"），
    空载不探测。首轮立即到期（_last_start=0），冷启动也能尽快拿到基线。

    探测跑在守护线程里，绝不阻塞派工主流程；一轮未结束不叠新一轮（_running 守卫）。
    探测只写延迟档案，**不碰**冷却与当日隔离：延迟是"快不快"的事实，改不了"能不能用"
    的判定（那是 health 的职责），否则定时器会变成自己给自己制造隔离的噪声源。
    """

    def __init__(self, store: LatencyStore, interval: float, probe_one, concurrency: int = 5,
                 clock=time.time):
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
        self._last_start = 0.0   # 0 = 从未探测 ⇒ 首次检查即到期
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
        threading.Thread(
            target=self.probe_all, args=(list(names),), daemon=True
        ).start()
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

    # ---------- 探测 ----------

    def probe_all(self, names) -> dict:
        """同步跑完一轮探测，返回 {工人: 毫秒 或 None}（外部主强制刷新 / 测试用）。

        每个工人独立探测，单个失败/超时不牵连其它工人；成功样本写进档案。
        探测失败**不**记 health / 冷却（见类文档）。
        """
        out: dict[str, float | None] = {}
        lock = threading.Lock()
        sem = threading.Semaphore(self._concurrency)
        waiters: list[threading.Thread] = []

        def _one(nm: str) -> None:
            with sem:
                try:
                    ms = self._probe_one(nm)
                except Exception:  # noqa: BLE001 - 单个工人探测异常不影响整轮
                    ms = None
            if ms is not None:
                try:
                    self._store.record(nm, ms)
                except Exception:  # noqa: BLE001 - 记账失败只丢持久化
                    pass
            with lock:
                out[nm] = ms

        for nm in dict.fromkeys(names):  # 去重且保序
            t = threading.Thread(target=_one, args=(nm,), daemon=True)
            waiters.append(t)
            t.start()
        for t in waiters:
            t.join()
        with self._lock:
            self._running = False
            self._last_finish = self._clock()
            self._rounds += 1
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
        # 探测专用短超时（秒）：探测客户端不沿用真实派工的 llm.timeout（120s）+ 重试，
        # 否则坏节点会把一次测速拖成分钟级、更会挂住同步 probe 整轮（见 _probe_client_for）。
        self._latency_probe_timeout = cfg.latency_probe_timeout
        # 探测器常驻（外部主的 probe_latency_now 随时可用），但**自动**触发由 latency_probe
        # 把关：每 latency_probe_interval 秒对全部工人（含当前不可用节点）打一次极短往返
        #（用户口径：每 10 分钟自动探测一次所有节点）。关掉它 = 不自动探测，
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
        return ordered[:max(1, len(ordered) - drop)]

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
        gate = self._station_gate((p.base_url, p.api_key))
        if not gate.acquire(timeout=_PROBE_GATE_WAIT):
            return None  # 该站正在跑真实请求 ⇒ 本轮让路，不污染样本也不挂住探测轮
        try:
            t0 = time.monotonic()
            resp = self._probe_client_for(p).chat([{"role": "user", "content": _PROBE_PROMPT}])
            if not str(resp.get("content") or "").strip():
                return None  # 空回复不算有效往返，避免污染样本
            return (time.monotonic() - t0) * 1000.0
        finally:
            gate.release()

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
        不必等自动探测的 10 分钟周期。与自动探测同一份档案、同一套测量口径。
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
    with _shared_lock:
        inst = _shared_instances.get(key)
        if inst is None:
            inst = LatencyStore(key)
            _shared_instances[key] = inst
        return inst
