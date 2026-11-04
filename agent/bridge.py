"""外部智能体驱动模式（bridge）：谁启动并驱动本项目，谁就是主智能体。

本项目为驱动者（外部主 AI）提供三类能力，围绕一个目标——**用多个免费 key 帮主 AI
干脏活、并行/交叉验证，省主 AI 的 token**：
1. 本地工具执行（shell / 文件 / MCP 工具 / Skill）——给被沙箱限制的主一双"手"。
2. 按需调用池内免费模型当"纯文本子 AI"：ask（单个）/ ask_many（并行+结构化）/
   ask_vote（两步投票取共识）/ run_review（主给初稿、子 AI 只当评审团挑错）。
3. 派工前预检：health（各子 AI 冷却/当日隔离/能力标签/延迟）、list_agents（池清单）、
   latency（延迟档案 + 探测状态，可强制刷一轮）。

协议：stdin 每行一个 JSON 请求，stdout 每行一个 JSON 响应，UTF-8；启动即发一行 ready。
诊断/警告一律走 stderr，stdout 只放 JSON 行，供外部主按行 json.loads 解析。

请求指令：
  {"cmd":"ping"}
  {"cmd":"list_agents"}                                   （池清单，含能力标签，不含 key）
  {"cmd":"health"}                                        （各子 AI 可用性快照：冷却/当日隔离/标签/延迟ms）
  {"cmd":"latency"}                                       （延迟档案 + 探测调度状态）
  {"cmd":"latency","probe":true}                          （先同步探测一轮全部节点再回包）
  {"cmd":"set_economy","value":true}                      （运行期切"省 token"开关，返回新状态）
  {"cmd":"refresh_nodes"}                                 （强制跑一轮免费节点发现：新增/下架自动适配，热同步进池）
  {"cmd":"list_tools"}
  {"cmd":"call_tool","name":"run_shell","args":{"command":"dir"}}
  {"cmd":"load_skill","name":"example_hello"}
  {"cmd":"run_skill_script","skill":"example_hello","script":"hello.py","args":{}}
  {"cmd":"ask","agent":"glm-flash","prompt":"...","system":"可选角色"}
  {"cmd":"ask_many","workers":["a","b"],"prompt":"..."}    （并行派工，返回 workers 结构化 + results 文本）
  {"cmd":"ask_vote","prompt":"...","threshold":0.5}         （两步投票取共识）
  {"cmd":"run_review","draft":"...","context":"可选","workers":["a","b"]}  （外部主给初稿，子 AI 只当评审团）

响应：{"ok":true,...} 或 {"ok":false,"error":"..."}。
  · ok 一律反映真实成败（#2/#3）：
    - call_tool / ask：工具或子 AI 失败即 ok:false；
    - ask_many / run_review：ok = 至少一个工人成功（逐工人结果见 workers/reviews）；
    - ask_vote：ok = 投票是否真正产出结果，另带 consensus 布尔（是否过半）。
  · ask / ask_many 的结构化条目带稳定 status 码：ok|error|cooldown|quarantined|missing|timeout。
"""

from __future__ import annotations

import json
import sys

from . import __version__
from .config import Config
from .core.orchestrator import WorkerPool
from .core.redact import mask_secrets, safe_error_text
from .skills_manager import SkillManager, register_skill_tools
from .tools.base import ToolRegistry
from .tools.builtin import build_builtin_tools

REVIEW_SYSTEM = (
    "你是独立评审智能体。针对给你的稿件只挑问题、给改进方向与编号建议，"
    "不改写全文、不泛泛夸赞。若整体可用就明确说'可用'并列出仅剩的小问题。"
)


def _norm_workers(raw) -> tuple[list[str] | None, str | None]:
    """把请求里的 workers 字段规范成 (名单, 错误)。

    三种入参必须区分开，否则"调用方明确不要任何人"会被当成"没点名"：
    - 字段缺失/为 None → (None, None)：走 `pick()` 的自动选路；
    - 显式空名单（`[]` / `""` / `"  "`）→ (None, 错误文案)：直接报错。旧写法在这里
      折叠成 None，于是 `{"workers": []}` 变成"全池轮转派工"，与调用方意图相反，
      还白烧一整轮额度（vote 更是要收集 + 表决两轮）；
    - 非空名单 → (名单, None)。
    """
    if raw is None:
        return None, None
    if isinstance(raw, str):
        names = [w.strip() for w in raw.split(",") if w.strip()]
    elif isinstance(raw, (list, tuple, set)):
        names = [str(w).strip() for w in raw if str(w).strip()]
    else:
        return None, "workers 必须是字符串或字符串列表"
    if not names:
        return (
            None,
            "workers 显式传空：要么点名，要么省略该字段让池子自动挑（不能既写空又指望全池）",
        )
    return names, None


def _parse_economy(raw) -> bool | None:
    """把 set_economy 的 value 容错解析成 bool；非法值返回 None（调用方保持原值）。"""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
    return None


def _norm_timeout(raw) -> float | None:
    """把请求里的 timeout 规范成秒数；缺省/非法 → None（让内核按规模自适应）。

    必须过一遍再传：内核把它当"总预算"直接 `float(timeout)`，外部主传 `"120"` 还能蒙对，
    传 `0`/`"很多"` 就分别是"限时 0 秒（全员判未完成）"与 TypeError。与数值配置同一态度：
    非法值警告一次并回退自适应，不阻塞派工。
    """
    if raw is None or raw == "":
        return None
    val = Config.as_float(raw, "request.timeout", 0.0, minimum=0.0)
    if val <= 0:
        print(f"[bridge] timeout={raw!r} 不是正秒数，已回退自适应限时", file=sys.stderr)
        return None
    return val


def _warn_oversized_named_list(cfg: Config, workers: list[str] | None) -> None:
    """显式点名超过 max_participants 时提醒一次（stderr，不阻断派工）。

    显式名单按约定不受该上限约束（调用方自己定规模），但"点 40 个 = 40 次真实派工"
    是额度维度的事实：池大 + 并发低时尾部批次还会撞整体限时被判未完成。
    这里只如实提示、把决定权留给主——静默放大额度比多一行 stderr 更糟。
    """
    if not workers:
        return
    cap = cfg.max_participants
    if cap > 0 and len(workers) > cap:
        print(
            f"[bridge] 显式点名 {len(workers)} 个工人，超过 collaboration.max_participants={cap}；"
            "显式名单不受该上限约束，本次仍按点名全部派工（额度消耗按人数计）",
            file=sys.stderr,
        )


class Bridge:
    """一个 bridge 连接：共享的注册表 + 技能 + MCP 连接组 + 工人池。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.skills = SkillManager()
        self.skills.scan()
        self.registry: ToolRegistry = build_builtin_tools(cfg)
        register_skill_tools(self.registry, self.skills)
        # MCP 走进程内共享连接组：同进程只维护一套连接，避免重复拉起 stdio 子进程、
        # 重复等最长 90s 的首连。
        self.mcp_group = None
        self.mcp_status: list[str] = []
        if cfg.mcp_servers:
            from .tools.mcp_client import acquire_group, register_group_tools

            self.mcp_group = acquire_group(cfg.mcp_servers, cfg.mcp_tool_timeout)
            self.mcp_status = list(self.mcp_group.status)
            register_group_tools(self.registry, self.mcp_group)
        self.workers = WorkerPool(cfg)
        # 省 token 开关（运行态）：初始值来自 collaboration.economy（.env COLLAB_ECONOMY）。
        # 外部主可经 set_economy 指令运行期切换；MAO 只透传开关、不做模式判断。
        self._economy: bool = cfg.economy

    # ---------- 指令分发 ----------

    def handle(self, req: dict) -> dict:
        """处理一条请求，返回最终响应 dict（serve 循环负责写 stdout）。"""
        cmd = req.get("cmd")
        if cmd == "ping":
            return {
                "ok": True,
                "version": __version__,
                # 读**活池**而不是 cfg.agent_profiles：热同步（refresh_nodes）只改工人池，
                # 静态配置不会跟着变，主若按这份清单点名就会撞到"不存在"的工人。
                "agents": list(self.workers.profiles),
            }
        if cmd == "list_agents":
            return {"ok": True, "agents": [p.brief() for p in self.workers.profiles.values()]}
        if cmd == "list_tools":
            return {"ok": True, "tools": self.registry.info_list()}
        if cmd == "health":
            # 每个子 AI 的当前可用性快照（冷却/当日隔离/能力标签/延迟ms），供外部主派工前预检。
            # 顺带回显 economy 开关：外部主每次派工前都无需记忆上次 set 的状态。
            # 再带回 concurrency（同站串行后的**真实**并行度）：点名一批工人能并行几路不再
            # 等于 max_workers，主据此决定"一批几个、分几轮、要不要跨站点名"。
            out = {
                "ok": True,
                "workers": self.workers.health_status(),
                "economy": self._economy,
                "concurrency": self.workers.concurrency_status(),
            }
            return out
        if cmd == "latency":
            # 延迟维度快照（问题5）：各工人当前延迟（往返耗时中位数，毫秒）+ 探测调度状态。
            # probe=true ⇒ 先**同步**跑一轮全节点探测再回包：外部主刚加了节点、或怀疑
            # 某模型变慢时可立即刷新，不必等 10 分钟自动周期。探测失败不打 health/冷却。
            if req.get("probe"):
                self.workers.probe_latency_now()
            return {"ok": True, **self.workers.latency_status()}
        if cmd == "set_economy":
            # 运行期切"省 token"开关（二元）。非法值警告并保持原值：
            # 返回 ok:false + error + 当前真实状态，调用方据此知道没切动。
            val = _parse_economy(req.get("value"))
            if val is None:
                print(
                    f"[bridge] set_economy 收到非法值 {req.get('value')!r}，保持当前 {self._economy}",
                    file=sys.stderr,
                )
                return {
                    "ok": False,
                    "economy": self._economy,
                    "error": "value 必须是布尔值或 true/false/1/0 等字面量",
                }
            self._economy = val
            return {"ok": True, "economy": self._economy}
        if cmd == "refresh_nodes":
            # 手动强制跑一轮免费节点发现：拉各站在线免费清单 → 对账 → 写叠加层
            # data/free_nodes.json，并热同步进当前运行中的工人池（不重启即生效）。
            # 增量语义：新增免费模型进池、在线消失的移出池、人工 # 屏蔽的永不自动复活。
            return {"ok": True, **self.workers.refresh_free_nodes_now()}
        if cmd == "call_tool":
            name = str(req.get("name", ""))
            res = self.registry.run(name, req.get("args") or {})
            # 工具正文（成功与失败同一份）在这里进 stdout 行协议 → 外发给另一个模型。
            # run_shell 已在源头掩码，这一层是**所有**工具的公共出口（MCP 远程工具、
            # 技能脚本、read_file 读到的配置文件正文都可能带 key），不逐个源头补。
            text = mask_secrets(res["result"])
            out = {"ok": res["ok"], "result": text}
            if not res["ok"]:
                out["error"] = text
            return out
        if cmd == "load_skill":
            content = self.skills.load_full(str(req.get("name", "")))
            ok = bool(getattr(content, "ok", True))
            text = mask_secrets(str(content))
            out = {"ok": ok, "content": text}
            if not ok:
                out["error"] = text
            return out
        if cmd == "run_skill_script":
            res = self.skills.run_script(
                str(req.get("skill", "")), str(req.get("script", "")), req.get("args") or {}
            )
            ok = bool(getattr(res, "ok", True))
            text = mask_secrets(str(res))
            out = {"ok": ok, "result": text}
            if not ok:
                out["error"] = text
            return out
        if cmd == "ask":
            prompt = str(req.get("prompt", "")).strip()
            if not prompt:
                return {"ok": False, "error": "prompt 不能为空"}
            agent = str(req.get("agent", "")).strip()
            if agent:
                r = self.workers.ask_result(agent, prompt, req.get("system"))
                # 统一契约：answer 只放成功正文，失败信息进 error/status（稳定码，不翻译）
                if r["ok"]:
                    return {"ok": True, "answer": r["answer"], "status": r["status"]}
                return {"ok": False, "answer": "", "status": r["status"], "error": r["error"]}
            # 未指定 agent：从池中挑一个当前可用工人（即"主模型"充当兜底），
            # 复用工人池的健康/冷却/回退链，不再需要单独的兜底单模型配置。
            # tags：内核一直支持按能力标签选路，但协议层原先从不透传（文档却写了这个参数）。
            pool = self.workers.pick(limit=1, tags=req.get("tags"))
            if not pool:
                return {
                    "ok": False,
                    "status": "missing",
                    "error": "当前没有可用工人（池全挂/全冷却），请指定 agent 或稍后再试",
                }
            r = self.workers.ask_result(pool[0], prompt, req.get("system"))
            if r["ok"]:
                return {"ok": True, "answer": r["answer"], "status": r["status"]}
            return {"ok": False, "answer": "", "status": r["status"], "error": r["error"]}
        if cmd == "ask_many":
            prompt = str(req.get("prompt", "")).strip()
            if not prompt:
                return {"ok": False, "error": "prompt 不能为空"}
            # 一次收集，同时回结构化（外部主程序化消费）与文本（人读）
            named, workers_err = _norm_workers(req.get("workers"))
            if workers_err:
                return {"ok": False, "error": f"ask_many：{workers_err}"}
            _warn_oversized_named_list(self.cfg, named)
            recs = self.workers.ask_many_structured(
                named if named is not None else self.workers.pick(tags=req.get("tags")),
                prompt,
                req.get("system"),
                timeout=_norm_timeout(req.get("timeout")),
            )
            text = "\n\n".join(
                f"### 工人 {r['worker']} 的结果\n"
                + ((r["answer"] or "（空正文）") if r["ok"] else f"（{r['status']}）")
                for r in recs
            )
            # #2：外层 ok = 至少一个工人成功；全失败/无可用工人 → ok:false
            usable = any(r["ok"] for r in recs)
            out = {"ok": usable, "results": text, "workers": recs}
            if not usable:
                out["error"] = "全部子 AI 均未成功（或无可用工人）"
            return out
        if cmd == "ask_vote":
            prompt = str(req.get("prompt", "")).strip()
            if not prompt:
                return {"ok": False, "error": "prompt 不能为空"}
            named, workers_err = _norm_workers(req.get("workers"))
            if workers_err:
                return {"ok": False, "error": f"ask_vote：{workers_err}"}
            _warn_oversized_named_list(self.cfg, named)
            v = self.workers.vote(
                prompt,
                named,
                req.get("threshold"),
                timeout=_norm_timeout(req.get("timeout")),
                master_contribution=req.get("master_contribution"),
            )
            # #2：vote 返回结构化 {ok, consensus, report}——ok=投票是否真正产出结果，
            # consensus=是否过半；无法投票时 ok:false 且带 error。
            out = {"ok": v["ok"], "result": v["report"], "consensus": v["consensus"]}
            if not v["ok"]:
                out["error"] = v["report"]
            return out
        if cmd == "run_review":
            draft = str(req.get("draft", "")).strip()
            if not draft:
                return {"ok": False, "error": "draft 不能为空（外部主给出初稿，子 AI 只当评审团）"}
            context = str(req.get("context", "")).strip()
            named, workers_err = _norm_workers(req.get("workers"))
            if workers_err:
                return {"ok": False, "error": f"run_review：{workers_err}"}
            _warn_oversized_named_list(self.cfg, named)
            reviewers = named if named is not None else self.workers.pick(tags=req.get("tags"))
            prompt = (
                "请作为独立评审，针对下面的稿件给出问题清单与**编号**改进建议，"
                "直接挑错、给方向，不要客套、不要整段复述原文。\n"
                + (f"【背景 / 验收标准】{context}\n" if context else "")
                + f"【待评审稿件】\n{draft}"
            )
            recs = self.workers.ask_many_structured(
                reviewers, prompt, REVIEW_SYSTEM, timeout=_norm_timeout(req.get("timeout"))
            )
            usable = any(r["ok"] for r in recs)
            out = {"ok": usable, "reviews": recs}
            if not usable:
                out["error"] = "全部评审工人均未成功（或无可用工人）"
            return out
        return {"ok": False, "error": f"未知指令: {cmd}"}

    def close(self) -> None:
        """释放共享工人池与 MCP 连接组引用。"""
        # 工人池持有 httpx 连接池，进程退出前显式释放。
        # MCP 只解除本使用者的引用（引用归零时组内部才真正断开）。
        try:
            self.workers.close()
        except Exception:  # noqa: BLE001 - 收尾失败不影响进程退出
            pass
        if self.mcp_group is not None:
            from .tools.mcp_client import release_group

            release_group(self.mcp_group)
            self.mcp_group = None


def reconfigure_streams() -> None:
    """把三件套 stdio 统一为 UTF-8。

    Windows 上重定向的流默认走本地代码页（GBK）：bridge 声明协议是 UTF-8，
    而配置警告等诊断走 stderr，外部驱动方按 UTF-8 解码 stderr 时会直接
    UnicodeDecodeError。CLI 单独跑无碍，但协议模式下必须归一。
    由 main()（bridge 子命令、load_config 之前）与 serve() 双入口调用。

    `errors="backslashreplace"` 同样关键：只设 encoding 不设错误策略时，Windows 的
    stderr 默认 errors=strict，一条含 emoji/生僻字的诊断（模型回答里很常见）就会抛
    UnicodeEncodeError，把"写一行日志"变成"整条派工被判失败"。
    """
    for stream in (sys.stdout, sys.stdin, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:  # noqa: BLE001 - 老版本 Python 或非 tty 场景尽力而为
            pass


def serve(cfg: Config) -> None:
    """阻塞式协议循环：stdin 请求行 → stdout 响应行。

    并发语义（#7）：本 shell 逐行**串行**处理——一条指令（如自适应限时可达数分钟的
    ask_many）执行期间，后续行排队等待，且不支持中途取消；要并行只能多开进程。
    而 mcp 外壳下宿主可并发发工具调用、同步工具跑在线程池里，故内核共享态
    （工人池健康、客户端缓存、轮转游标等）均已按可并发访问设计/加锁。两者行为一致，
    差别只在"一次一条"还是"可多条同时在飞"。
    """
    reconfigure_streams()
    bridge = Bridge(cfg)
    ready = {
        "ok": True,
        "event": "ready",
        "version": __version__,
        "agents": list(bridge.workers.profiles),
        "mcp": bridge.mcp_status,
        "economy": bridge._economy,  # 省 token 开关初始值，外部主握手即可知当前模式
    }
    sys.stdout.write(json.dumps(ready, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                if isinstance(req, dict):
                    resp = bridge.handle(req)
                else:
                    resp = {"ok": False, "error": "请求必须是 JSON 对象"}
            except json.JSONDecodeError as e:
                # 统一走脱敏：这一行直接进 stdout 的 JSON 行协议，不留"哪个分支可以原文透传"的判断
                resp = {"ok": False, "error": f"JSON 解析失败: {safe_error_text(e)}"}
            except Exception as e:  # noqa: BLE001 - 单条指令失败不能拖垮协议循环
                # 这一行会直接写进 stdout 的 JSON 行协议，异常原文可能带 endpoint，先脱敏
                resp = {"ok": False, "error": f"{type(e).__name__}: {safe_error_text(e)}"}
            try:
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()
            except OSError as e:
                # 写不出这一行 = 宿主已经把管道关了/磁盘错误。静默崩栈会让"外部主为什么
                # 突然收不到回包"无从排查，故走 stderr 留一行再退出循环（bridge.close 照跑）。
                print(
                    f"[bridge] stdout 写入失败，协议循环退出："
                    f"{type(e).__name__}: {safe_error_text(e)}",
                    file=sys.stderr,
                )
                break
    finally:
        bridge.close()
