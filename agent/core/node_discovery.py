"""免费节点（免费档模型）自动发现与对账。

为什么需要：提供方的免费模型清单会随时间变化（新增/下架），原来纯靠手维护
model_registry.txt + 连续失败 7 天才自动下线，既发现不了**新增**的免费模型、也对
**下架**反应太慢（期间每只白打请求、烧额度）。本模块按"站"挂适配器，拉在线免费清单
→ 与 model_registry.txt 对账 → 把**增量**写进 data/free_nodes.json 叠加层（绝不改 git
跟踪的 model_registry.txt）：

  added   = 在线有、registry 里完全没有的模型 → 追加入池
  retired = registry 活跃、在线消失的模型     → 从池剔除（不改 registry 行）

三条铁律：
1. 人工 `#` 屏蔽的模型永不自动复活（registry 里带 # 的项，bare 名仍计入"已存在"，
   不会被 added 重新加回，也不会被 retired 误删）。
2. 整站抓取失败（网络/鉴权/解析异常）→ 该站跳过、不动 overlay → 原 registry 生效，
   离线/测试零影响。
3. 增量相对 **registry 原始行** 计算（不是相对"上次叠加后的结果"），因此多次运行幂等，
   不会把已加的模型重复叠加。

适配器：
- kilo   ：公开接口 `https://api.kilo.ai/api/gateway/models`（无需登录），`id` 以
          `:free` 结尾即免费档，带 `context_length`。

bynara（node-a）暂未接入自动发现：其 `/v1/models` 端点只认 API Key Bearer、且免费模型
字段结构需要真实返回样本才能定稿解析器（网站 cookie 实测 401、拿不到样本），故 node-a
退回静态站——免费模型变化沿用本文档手写更新 + 连续失败 7 天自动下线，与其它静态站一致。

触发（与延迟探测同一套"懒式"调度）：建池时 best-effort 探一次 + 每 free_node_interval
秒后台探一轮 + bridge `refresh_nodes` 手动强制。叠加层在**下次 load_config** 生效，
`refresh_nodes` 同时把增量热同步进当前运行中的工人池（见 WorkerPool.sync_free_nodes）。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from ..model_registry import load_model_registry
from .discovery_adapters import (  # noqa: F401 - re-export：历史导入面不变
    _MODEL_ID_RE,
    Adapter,
    FreeModel,
    KiloAdapter,
    _as_int,
    _clean_models,
    _http_get_json,
    _manages,
    _owns_station,
)
from .ledger import _warn_write_failed, atomic_write_text, file_lock

_DEFAULT_OVERLAY = "data/free_nodes.json"


# ---------- 基础解析工具（与 model_registry.py 的 _bare_model 同口径） ----------

_SIZE_SUFFIX = re.compile(r"\d+[kK]?")


def _bare(item: str) -> str:
    """剥掉 # 前缀与 @窗口标注得到 bare 模型名：`#laguna-s-2.1@256k` → `laguna-s-2.1`。

    与 config._split_models / model_registry._bare_model 同一套"什么算窗口标注"规则：
    只有 `@` 之后是纯数字（可带 k/K）才算窗口，否则 @ 属于模型名（如 Cloudflare 的
    `@cf/...`）。
    """
    body = item.lstrip("#")
    name, sep, suffix = body.rpartition("@")
    bare = name.strip() if sep and _SIZE_SUFFIX.fullmatch(suffix.strip()) else body.strip()
    return ("#" if item.startswith("#") else "") + bare


def _split_items(raw: str) -> list[str]:
    return [p.strip() for p in str(raw or "").split(",") if p.strip()]


def _parse_one(entry: str) -> tuple[str, int]:
    """把 `name@ctx` 单条拆成 (bare 名, 窗口)。窗口缺省 0。"""
    item = entry.strip()
    name, sep, suffix = item.rpartition("@")
    if sep and _SIZE_SUFFIX.fullmatch(suffix.strip()):
        s = suffix.strip().lower()
        mul = 1024 if s.endswith("k") else 1
        num = s[:-1] if s.endswith("k") else s
        try:
            return name.strip(), int(num) * mul
        except ValueError:
            return name.strip(), 0
    return item, 0


def _format_model(m: FreeModel) -> str:
    """叠加层里存储的条目串：有窗口就带 `@ctx`（原样存 int，config 解析兼容）。"""
    return f"{m.model}@{m.ctx}" if m.ctx else m.model


# ---------- 适配器（每站一个，知道怎么拉+怎么解析） ----------


# ---------- 叠加层读写 ----------


def load_overlay(path: Path) -> dict:
    """读 data/free_nodes.json；缺失/损坏返回空 dict（不抛异常）。"""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def save_overlay(path: Path, data: dict) -> bool:
    """原子写叠加层；返回是否真正落盘成功。"""
    try:
        atomic_write_text(Path(path), json.dumps(data, ensure_ascii=False, indent=2))
        return True
    except OSError as e:
        _warn_write_failed(path, e)
        return False


def merge_delta(base: str, delta: dict) -> str:
    """把一条叠加层增量应用到 registry 原始行上，返回新行。

    - 剔除 retired（仅从活跃项移除；手动 # 项本来不在活跃集，保留）；
    - 追加 added（bare 名已存在则跳过，**含手动 # 屏蔽的**，保证不自动复活）。
    比较一律用"去 # 前缀的 bare 名"，否则手动 `#m1` 与 added `m1` 会被当成两个不同模型、
    导致被屏蔽的模型被悄悄复活。纯字符串操作，无副作用。
    """
    items = _split_items(base)
    retired = {_bare(r).lstrip("#") for r in (delta.get("retired") or [])}
    kept = [i for i in items if _bare(i).lstrip("#") not in retired]
    present = {_bare(i).lstrip("#") for i in kept}
    for a in delta.get("added") or []:
        if _bare(a).lstrip("#") in present:
            continue
        kept.append(a)
        present.add(_bare(a).lstrip("#"))
    return ",".join(kept)


def apply_overlay(overlay_path: Path) -> None:
    """在 load_config 注入 dotenv 之后调用：把叠加层增量叠加到对应 env 变量上。

    纯本地文件读取（无网络），叠加层存在就应用——与 MAO_FREE_NODE_PROBE 无关：
    该开关只禁止"联网去拉新清单"，不禁止"套用已发现的结果"。无叠加层文件时整段跳过。
    只改"确实存在于 os.environ 的 NODE_*_MODELS 变量"，不动其它、不动 model_registry.txt。
    """
    data = load_overlay(overlay_path)
    if not data:
        return

    for env_var, delta in data.items():
        if env_var.startswith("_") or not isinstance(delta, dict):
            continue
        if env_var not in os.environ:
            continue
        patched = merge_delta(os.environ[env_var], delta)
        if patched != os.environ[env_var]:
            os.environ[env_var] = patched


# ---------- 发现器（含懒式调度，复用 LatencyProber 思路） ----------


def _reconcile(
    registry_line: str,
    live: list[FreeModel],
    adapter: Adapter | None = None,
    previous: dict | None = None,
) -> dict:
    """相对 registry 与上一轮自动新增项计算当前快照，确保新增/下线都可撤销。"""
    items = _split_items(registry_line)
    registry_active = {_bare(i).lstrip("#") for i in items if not i.startswith("#")}
    registry_all = {_bare(i).lstrip("#") for i in items}
    previous_added = {
        _bare(a).lstrip("#") for a in ((previous or {}).get("added") or []) if isinstance(a, str)
    }
    live_bare = {m.model for m in live}
    added = [_format_model(m) for m in live if m.model not in registry_all]
    retired = [
        m
        for m in registry_active | previous_added
        if m not in live_bare and (adapter is None or _manages(adapter, m))
    ]
    return {"added": added, "retired": retired}


def _delta_signature(d: Any) -> tuple[frozenset, frozenset]:
    """增量的集合签名：added / retired 各自按 bare 名取 frozenset。

    为什么要"集合比较"而不是 list 相等（2026-11-04 审计 P2）：added 的顺序跟远端清单
    顺序走、retired 来自 set 迭代（跨进程 hash 种子不同则顺序不同）——远端只是顺序抖动
    一下，list 比较就判"有变化"，叠加层被整篇重写、sync_free_nodes 白跑。集合相同
    就不算变化。
    """
    if not isinstance(d, dict):
        return frozenset(), frozenset()
    return (
        frozenset(_bare(a).lstrip("#") for a in d.get("added") or []),
        frozenset(_bare(r).lstrip("#") for r in d.get("retired") or []),
    )


def _merge_deltas(prev: Any, delta: dict) -> dict:
    """把本轮增量合并进盘上已有增量：added 保序取并集，retired 排序取并集。

    服务跨进程写前重读（见 run_once）：两个进程各自 load→改→save 全文时，后写者
    必须把先写者的**其它站增量**原样保留；同一站两轮结果合并时取并集（幂等）。
    retired 存为有序列表（排序后跨进程稳定），比较走 _delta_signature 不受顺序影响。
    """
    base = prev if isinstance(prev, dict) else {}
    added: list[str] = []
    seen: set[str] = set()
    for a in list(base.get("added") or []) + list(delta.get("added") or []):
        bare = _bare(a).lstrip("#")
        if bare in seen:
            continue
        seen.add(bare)
        added.append(a)
    retired = sorted(
        {_bare(r).lstrip("#") for r in base.get("retired") or []}
        | {_bare(r).lstrip("#") for r in delta.get("retired") or []}
    )
    return {"added": added, "retired": retired}


class FreeNodeDiscoverer:
    """按站挂适配器，拉在线免费清单并与 registry 对账，写叠加层。

    懒式调度（与延迟探测同思路）：常驻外壳 CLI 一次调用即退出，故不自建常驻定时线程，
    而是挂在派工/预检入口顺带检查是否到期；定时轮在守护线程里跑，绝不阻塞主流程，
    且同一时刻只允许一轮在飞。free_node_interval<=0 或 free_node_discovery=false 或
    MAO_FREE_NODE_PROBE=0 → 完全不调度。
    """

    def __init__(self, cfg, overlay_path: Path, adapters: dict[str, Adapter] | None = None):
        self.cfg = cfg
        self.overlay_path = Path(overlay_path)
        self.adapters = adapters or {"kilo": KiloAdapter()}
        self._interval = max(0.0, float(cfg.free_node_interval))
        self._timeout = max(1.0, float(cfg.free_node_timeout))
        self._lock = threading.Lock()
        # 整轮互斥锁：自动轮与手动 refresh_nodes 都要过它，保证绝不会有两轮同时在读改写叠加层
        # （叠加层是 load → 改本站 → save 全文，两轮并发时后 save 的会把前一站的增量整段抹掉，
        # 与 ledger.py 记的那类 lost update 同源）。嵌套方向恒为 _round_lock → _lock。
        self._round_lock = threading.Lock()
        self._running = False
        self._last_start = 0.0
        self._last_finish = 0.0
        self._rounds = 0
        self._clock = time.time

    def enabled(self) -> bool:
        """是否允许**联网**发现（总开关：kill switch + free_node_discovery）。

        周期到没到点是 `due()` 的事，绝不混进这里：旧写法把 `_interval > 0` 也算成"未启用"，
        于是 `COLLAB_FREE_NODE_INTERVAL=0`（config.yaml 注释承诺"仅手动 refresh_nodes 可触发"）
        会把手动刷新一起废掉。
        """
        if os.environ.get("MAO_FREE_NODE_PROBE") == "0":
            return False
        return bool(self.cfg.free_node_discovery)

    def due(self) -> bool:
        """当前是否该起一轮**自动**发现（配了周期、到期且无在飞）。"""
        with self._lock:
            if self._interval <= 0 or self._running:
                return False
            return (self._clock() - self._last_start) >= self._interval

    def maybe_start(self) -> bool:
        """到期且无在飞 ⇒ 起守护线程跑一轮；返回是否真起了（调用方不等待）。

        到期判定与占位在**同一次持锁**内完成（2026-11-04 审计）：旧写法 due() 与占位
        分两次取锁，中间若被手动 refresh_now 插队跑完一轮（_last_start 已刷新），这里
        只复查 _running、不复查周期 ⇒ 立刻又起一轮冗余自动发现（多打一次网络）。
        """
        if not self.enabled():
            return False
        with self._lock:
            if self._running or self._interval <= 0:
                return False
            if (self._clock() - self._last_start) < self._interval:
                return False
            self._running = True
            self._last_start = self._clock()
        try:
            threading.Thread(target=self._run_thread, daemon=True).start()
        except (RuntimeError, OSError):
            # 线程起不来必须把周期起算点也退回上一轮结束时间：只复位 _running 的话，
            # _last_start 已被刷新 ⇒ 这一轮发现被丢掉，还要再等一整个 interval 才有机会
            # 重来（LatencyProber 的同款问题当初是靠 _release_claim 修的，这里补齐）。
            with self._lock:
                self._running = False
                self._last_start = self._last_finish
            return False
        return True

    def _run_thread(self) -> None:
        try:
            with self._round_lock:  # 与手动轮互斥：并发跑两轮会各自 load/save 同一叠加层
                self.run_once()
        except Exception:  # noqa: BLE001 - 发现失败绝不影响派工主流程
            pass
        finally:
            with self._lock:
                self._running = False
                self._last_finish = self._clock()
                self._rounds += 1

    def _targets(self) -> dict[str, dict]:
        """收集带 discovery 标记的 profile，按 models_env 归并（取 base_url/api_key）。"""
        by_env: dict[str, dict] = {}
        for p in self.cfg.agent_profiles:
            if p.discovery and p.models_env:
                info = by_env.setdefault(
                    p.models_env,
                    {"base_url": p.base_url, "api_key": p.api_key, "discovery": p.discovery},
                )
                info["base_url"] = p.base_url
                info["api_key"] = p.api_key
        return by_env

    def run_once(self, clock: Any = None) -> dict:
        """同步跑一轮发现：逐站拉取 → 对账 → 写叠加层。返回摘要（不含密钥）。

        写叠加层前持跨进程锁重读磁盘：每个成功抓取的站点以本轮最新快照替换自己的
        added/retired，其它站点原样保留。这样模型重新上线会撤掉 retired，自动新增模型
        下架会从 added 中消失，同时不会丢失其它进程刚写入的站点增量。
        """
        targets = self._targets()
        registry = load_model_registry(self.cfg.model_registry_path)
        report: dict[str, Any] = {"stations": {}, "updated": False}
        deltas: dict[str, dict] = {}
        previous_overlay = load_overlay(self.overlay_path)
        for env_var, info in targets.items():
            adapter = self.adapters.get(info["discovery"])
            if adapter is None:
                continue
            if not _owns_station(adapter, info["base_url"]):
                # 站不属于这个适配器 ⇒ 它那份清单对本站毫无意义（派工到本站不存在的模型）
                report["stations"][env_var] = {"status": "skip", "reason": "host_mismatch"}
                continue
            registry_line = registry.get(env_var, "")
            try:
                live = adapter.fetch(info["base_url"], info["api_key"], self._timeout)
            except Exception:  # noqa: BLE001 - 适配器异常 ⇒ 跳过该站
                live = None
            if live is None:
                report["stations"][env_var] = {"status": "skip", "reason": "fetch_failed"}
                continue
            # 远端字符串先清洗再对账：含逗号/空白的 id 进叠加层等于往 env 注入多条模型
            live = _clean_models(live)
            if not live:
                report["stations"][env_var] = {"status": "skip", "reason": "no_valid_models"}
                continue
            deltas[env_var] = _reconcile(
                registry_line, live, adapter, previous_overlay.get(env_var)
            )
            report["stations"][env_var] = {
                "status": "ok",
                "live": len({m.model for m in live}),
                "added": deltas[env_var]["added"],
                "retired": deltas[env_var]["retired"],
            }
        if deltas:
            try:
                with file_lock(self.overlay_path):
                    disk = load_overlay(self.overlay_path)
                    changed = False
                    for env_var, delta in deltas.items():
                        if _delta_signature(delta) != _delta_signature(disk.get(env_var)):
                            disk[env_var] = delta
                            changed = True
                    if changed:
                        report["updated"] = save_overlay(self.overlay_path, disk)
            except OSError as e:
                _warn_write_failed(self.overlay_path, e)
        return report

    def refresh_now(self) -> dict:
        """同步强制刷新一轮（bridge `refresh_nodes` 用）；与自动轮互斥，同一时刻只一轮。

        旧写法直接硬置 `_running=True` 就开跑、不等在飞的自动轮 —— 两轮并发各自
        `load_overlay` → 改自己那站 → `save_overlay` 全文，后写的把先写的站点增量整段抹掉
        （与 ledger.py 记的那类 lost update 同源）。现在过 `_round_lock` 排队。
        周期配成 0（只关自动调度）时手动刷新照常可用，符合 config.yaml 的承诺。
        """
        if not self.enabled():
            return {"enabled": False, "stations": {}}
        with self._round_lock:
            with self._lock:
                self._running = True
                self._last_start = self._clock()
            try:
                return self.run_once()
            finally:
                with self._lock:
                    self._running = False
                    self._last_finish = self._clock()
                    self._rounds += 1

    def status(self) -> dict:
        """调度状态快照（诊断用，不含密钥）。

        `enabled`=允许联网发现（总开关），`scheduled`=是否自动排轮（interval>0）：
        两者原先混在一个值里，导致"关周期"顺手把 `refresh_nodes` 也废了。
        """
        with self._lock:
            return {
                "enabled": self.enabled(),
                "scheduled": self._interval > 0,
                "interval_s": self._interval,
                "running": self._running,
                "rounds": self._rounds,
            }
