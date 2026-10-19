"""免费节点自动发现：把叠加层增量热同步进运行中的工人池。

从 orchestrator.py 拆出（2026-10-19）：orchestrator.py 单文件已达 880 行，远超仓库
「单文件 ≤500 行」约定（规则5）。免费节点发现相关四个方法（_touch_free_nodes /
refresh_free_nodes_now / sync_free_nodes / free_node_status）约 95 行，与「池状态 /
pick 轮转 / 健康记账」职责不同，独立成一个 Mixin。

实现方式：FreeNodeMixin。WorkerPool 继承它，Mixin 方法通过 self 访问 WorkerPool 的
cfg / _free_node / profiles / _model_of / _env_of / _station_models / _lock 等属性
与方法，调用签名完全不变（bridge/mcp/call 三外壳无感）。本类不单独实例化。
"""

from __future__ import annotations

from ..config import AgentProfile
from .node_discovery import _bare, _parse_one, load_overlay


class FreeNodeMixin:
    """免费节点发现能力，混入 WorkerPool 使用（本类不单独实例化）。"""

    # ---------- 免费节点自动发现 ----------

    def _touch_free_nodes(self) -> None:
        """派工/预检入口的顺带动作：发现到期就起一轮后台发现（不阻塞、不抛异常）。

        与 _touch_latency 同理：挂在入口而非常驻线程，避免无人派工时空转烧额度；
        free_node_discovery=false / interval<=0 / MAO_FREE_NODE_PROBE=0 时整段跳过。
        """
        if not self._free_node.enabled():
            return
        try:
            self._free_node.maybe_start()
        except Exception:  # noqa: BLE001 - 发现调度绝不拖累派工
            pass

    def refresh_free_nodes_now(self) -> dict:
        """同步强制跑一轮免费节点发现，并把增量热同步进当前工人池（不重启即生效）。

        供 bridge `refresh_nodes` 命令调用。返回 {stations, updated, applied} 摘要（不含密钥）；
        发现失败/被禁用的站不计入变动。applied 是本次对运行中池子实际增/删的工人名。
        """
        report = self._free_node.refresh_now()
        try:
            overlay = load_overlay(self.cfg.free_nodes_path)
            report["applied"] = self.sync_free_nodes(overlay)
        except Exception:  # noqa: BLE001 - 热同步失败不炸命令
            report["applied"] = {"error": "sync_failed"}
        return report

    def sync_free_nodes(self, overlay: dict) -> dict:
        """把叠加层增量热同步进当前运行中的工人池：新增的进池、消失的移出池。

        不改 registry、不重启。手动 `#` 屏蔽的模型不在 added 里（发现阶段已排除），
        故不会因同步被重新启用。返回本次实际变动的工人名（不含密钥）。
        """
        with self._lock:
            profiles = dict(self.profiles)
            # 先按 env 抓一个模板（base_url/api_key/discovery 来源），即便该站活跃模型全被
            # retired 移除也能据此把新增模型建出来——否则模板随 retired 一起被删就没法加了。
            templates: dict[str, AgentProfile] = {}
            for p in profiles.values():
                templates.setdefault(p.models_env, p)
            removed: list[str] = []
            for env_var, delta in overlay.items():
                if env_var.startswith("_") or not isinstance(delta, dict):
                    continue
                retired = {_bare(r).lstrip("#") for r in (delta.get("retired") or [])}
                if not retired:
                    continue
                for name, p in list(profiles.items()):
                    if p.models_env == env_var and _bare(p.model).lstrip("#") in retired:
                        profiles.pop(name, None)
                        removed.append(name)
            added: list[str] = []
            for env_var, delta in overlay.items():
                if env_var.startswith("_") or not isinstance(delta, dict):
                    continue
                tmpl = templates.get(env_var)
                if tmpl is None:
                    continue
                existing = {
                    _bare(p.model).lstrip("#") for p in profiles.values() if p.models_env == env_var
                }
                station = tmpl.name.split(":")[0]
                for entry in delta.get("added") or []:
                    model, ctx = _parse_one(entry)
                    if not model or model in existing:
                        continue
                    # 新增模型统一用 "站名:模型名"（与 config 多模型站命名一致）
                    name = f"{station}:{model}"
                    np = AgentProfile(
                        name=name,
                        base_url=tmpl.base_url,
                        api_key=tmpl.api_key,
                        model=model,
                        context_length=ctx,
                        note=tmpl.note,
                        models_env=env_var,
                        discovery=tmpl.discovery,
                    )
                    profiles[name] = np
                    existing.add(model)
                    added.append(name)
            if removed or added:
                self.profiles = profiles
                self._model_of = {n: p.model for n, p in profiles.items()}
                self._env_of = {n: p.models_env for n, p in profiles.items()}
                self._station_models = {}
                for nm, p in profiles.items():
                    self._station_models.setdefault(p.station_key, []).append(nm)
            # 池锁内只取快照：回收要拿 _clients_lock / _gates_lock，绝不能嵌套在池锁里
            live_stations = {p.station_key for p in profiles.values()}
        # 锁外回收被移出工人的连接池与已空的站门（见 orchestrator._retire_worker_clients）
        if removed:
            self._retire_worker_clients(removed, live_stations)
        return {"added": added, "removed": removed}

    def free_node_status(self) -> dict:
        """免费节点发现维度诊断快照（不含密钥）。"""
        return {"discovery": self._free_node.status()}
