"""模型健康档案：当日失败隔离 + 连续多个"运行日"失败自动下线（在 .env.example 中加 # 前缀）。

规则（对应用户需求「失败打当天标记不再使用；连续 7 个启动日失败就下线」）：
- 某模型一次"终态失败"（重试后仍失败的调用），就往档案里该模型的 fail_dates
  记一个当天日期。当天已记录 ⇒ 当天不再派工该模型（WorkerPool 的
  ask/ask_many/collect/vote 全部跳过）；已缓存的回答仍可用（缓存命中零成本，
  与节点健康无关）。
- "运行日"（active day）= 程序实际启动并工作过的自然日，记录在档案顶层的
  active_dates。周末没开机的日子既不算运行日、也不打断连击。加载档案时今天
  自动记为运行日；每次失败记账也把当天记为运行日。
- 取最近 retire_days 个运行日（默认 7，collaboration.retire_days 可调）做窗口，
  若该模型在这些天"每天都失败过" ⇒ 自动把它在 .env.example 对应 *_MODELS 行
  里的条目前加 #（下线），并在档案记 disabled 标记、stderr 提示。
  恢复采用人工方式：删掉 # 即重新上线（不自动复活，避免抖动期反复横跳）。

档案格式（collaboration.health_file，默认 data/model_health.json）：
    {"version": 2,
     "active_dates": ["2026-09-20", "2026-09-21", ...],
     "models": {"node-a:nex-n2.5-pro": {"fail_dates": ["2026-09-21", ...],
                                        "disabled": "2026-10-01" 或省略}}}
fail_dates 只保留最近 _MAX_DATES_KEPT 条、active_dates 只保留最近
_MAX_ACTIVE_KEPT 条，防止文件无界增长。旧版 version 1（无 active_dates）
可读：视为运行日未知，从当天重新开始计。

并发：自带独立锁（与 WorkerPool 的锁相互独立，嵌套方向恒为 pool→health，
不会反向，无死锁环）；写盘原子化（临时文件 + os.replace）。
离线可测：record_failure 支持显式指定日期 day，不必等真实时间流逝。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

_MAX_DATES_KEPT = 30
_MAX_ACTIVE_KEPT = 60
_ENV_LINE = re.compile(r"^([A-Za-z0-9_]+)=(.*)$")


def _today() -> str:
    return time.strftime("%Y-%m-%d")


class ModelHealth:
    """模型健康账本：当日隔离查询 + 失败记账 + 连续运行日达阈值改写 .env.example 下线。"""

    def __init__(
        self,
        store_path: Path,
        env_example_path: Path | None = None,
        retire_days: int = 7,
    ):
        self.store_path = Path(store_path)
        self.env_example_path = Path(env_example_path) if env_example_path else None
        self.retire_days = max(1, int(retire_days))
        self._lock = threading.Lock()
        self._data: dict[str, dict] | None = None  # 懒加载：首次使用时才读盘
        self._active: list[str] = []  # 运行日清单（加载后随记账增长）

    # ---------- 查询 ----------

    def quarantined(self, name: str) -> bool:
        """当天记过失败 ⇒ 隔离（当天不再派工）。只查内存，热路径友好。"""
        with self._lock:
            self._ensure_loaded()
            entry = self._data.get(name) or {}
            return _today() in (entry.get("fail_dates") or [])

    # ---------- 记账 ----------

    def record_failure(self, name: str, model: str = "", models_env: str = "",
                       day: str | None = None) -> str | None:
        """记一次终态失败。返回非 None 时为下线提示文本（调用方负责打印 stderr）。

        name       档案键（通常是 AgentProfile.name，主智能体兜底模型用 solo:<model>）
        model      模型名（改写 .env.example 时按 @ 前的名字精确匹配条目）
        models_env 该模型清单所在的环境变量名（如 NODE_A_MODELS），优先改写此行
        day        记账日期（缺省=今天；测试可注入历史日期构造连击）
        """
        day = day or _today()
        notice: str | None = None
        with self._lock:
            self._ensure_loaded()
            self._mark_active_locked(day)  # 有失败记账 ⇒ 当天必然是运行日
            entry = self._data.setdefault(name, {})
            dates = sorted(set(entry.get("fail_dates") or []) | {day})
            entry["fail_dates"] = dates[-_MAX_DATES_KEPT:]
            if not entry.get("disabled") and self._streak_full_locked(entry["fail_dates"]):
                entry["disabled"] = day
                hit = self._retire_in_env_example(model, models_env)
                if hit:
                    notice = (
                        f"[模型健康] '{name}' 已连续 {self.retire_days} 个运行日调用失败"
                        "（未启动的日子不计入、也不打断），"
                        f"模型 '{model}' 已在 .env.example 的 {hit} 行标 # 下线；"
                        f"确认恢复免费额度/仍想使用时删掉 # 即可。"
                    )
                else:
                    notice = (
                        f"[模型健康] '{name}' 已连续 {self.retire_days} 个运行日调用失败，已记入档案，"
                        f"但在 .env.example 未找到模型 '{model}' 对应条目，请手动确认清单。"
                    )
            self._persist()
        return notice

    def note_active(self, day: str | None = None) -> None:
        """显式登记一个运行日（测试注入用：模拟"这天开过程序、但该模型没失败"）。"""
        with self._lock:
            self._ensure_loaded()
            self._mark_active_locked(day or _today())
            self._persist()

    # ---------- 运行日连击判定 ----------

    def _mark_active_locked(self, day: str) -> None:  # 调用方持锁
        if day not in self._active:
            self._active = sorted(set(self._active) | {day})[-_MAX_ACTIVE_KEPT:]

    def _streak_full_locked(self, fail_dates: list[str]) -> bool:
        """最近 retire_days 个运行日是否全部落在该模型的失败日里。

        运行日不足 retire_days 个时不判定下线（程序刚上手，样本不够）。
        周末没开机 ⇒ 那天不在 active_dates 里，既不计失败也不打断连击。
        """
        if len(self._active) < self.retire_days:
            return False
        window = self._active[-self.retire_days:]
        dset = set(fail_dates)
        return all(d in dset for d in window)

    # ---------- .env.example 改写（# 下线） ----------

    def _retire_in_env_example(self, model: str, models_env: str) -> str | None:
        """把 model 对应条目加 # 前缀，命中的环境变量名（可能多行）逗号连接返回；未命中 None。

        匹配范围：models_env 精确行优先，同时兼容任意 *_MODELS 行
        （同模型多站共用时一起下线，语义为「该上游模型不健康」）。
        不扫 generic *_MODEL：LLM_MODEL 是 solo 兜底单模型，被自动注释会让
        兜底路径直接失效，只允许人工下线。
        已有 # 前缀的条目视为已处理（幂等），不算本轮改动。
        整行按 ',' 拆条目、只对 '@' 前名字全等匹配的条目加 #，其余字节原样保留
        （含 CRLF：newline='' 读、按 '\n' 拆、原样拼回）。
        """
        if not model or self.env_example_path is None:
            return None
        try:
            with self.env_example_path.open("r", encoding="utf-8", newline="") as f:
                raw = f.read()
        except OSError:
            return None
        matched: list[str] = []
        out_lines: list[str] = []
        changed = False
        for line in raw.split("\n"):
            cr = line.endswith("\r")
            body = line[:-1] if cr else line
            m = _ENV_LINE.match(body)
            if m:
                var, val = m.group(1), m.group(2)
                candidate = var == models_env or var.endswith("_MODELS")
                if candidate:
                    entries = val.split(",")
                    hit_line = False
                    for i, e in enumerate(entries):
                        es = e.strip()
                        if not es:
                            continue
                        if es.startswith("#"):
                            # 已屏蔽：同模型也算命中（避免再兜底扫其它行），但不算改动
                            if es[1:].partition("@")[0].strip() == model:
                                hit_line = True
                            continue
                        if es.partition("@")[0].strip() == model:
                            entries[i] = "#" + es
                            hit_line = changed = True
                    if hit_line:
                        matched.append(var)
                        body = f"{var}=" + ",".join(entries)
            out_lines.append(body + ("\r" if cr else ""))
        if not changed:
            return ",".join(dict.fromkeys(matched)) or None
        try:
            tmp = self.env_example_path.with_name(self.env_example_path.name + ".tmp")
            with tmp.open("w", encoding="utf-8", newline="") as f:
                f.write("\n".join(out_lines))
            os.replace(tmp, self.env_example_path)
        except OSError:
            return None  # 写不回去就别谎报下线；档案里 disabled 已记，人工处理
        return ",".join(dict.fromkeys(matched)) or None

    # ---------- 落盘 ----------

    def _ensure_loaded(self) -> None:  # 调用方持锁
        if self._data is not None:
            return
        data: dict[str, dict] = {}
        active: list[str] = []
        try:
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
            models = raw.get("models") if isinstance(raw, dict) else None
            if isinstance(models, dict):
                data = {k: v for k, v in models.items() if isinstance(v, dict)}
            if isinstance(raw, dict):
                active = [d for d in (raw.get("active_dates") or []) if isinstance(d, str)]
        except (OSError, ValueError):
            data = {}  # 缺失/损坏都从空白账本重新开始，不拖垮派工
        self._data = data
        self._active = sorted(set(active))[-_MAX_ACTIVE_KEPT:]
        # 能加载到这一步说明程序今天在工作：当天先记为运行日。
        # 不立刻写盘（查询路径也走这里，避免只读操作产生副作用），
        # 由后续记账/登记的 _persist 一并落盘。
        self._mark_active_locked(_today())

    def _persist(self) -> None:  # 调用方持锁
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.store_path.with_name(self.store_path.name + ".tmp")
            payload = {"version": 2, "active_dates": self._active, "models": self._data}
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.store_path)
        except OSError:
            pass  # 档案写不进去只损失持久化，内存账本照常工作

    def snapshot(self) -> dict[str, dict]:
        """当前档案的浅拷贝（诊断/测试用，不含任何密钥）。"""
        with self._lock:
            self._ensure_loaded()
            return {k: dict(v) for k, v in self._data.items()}
