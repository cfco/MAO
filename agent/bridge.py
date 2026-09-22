"""外部智能体驱动模式（bridge）：谁启动并驱动本项目，谁就是主智能体。

本项目为驱动者（主智能体）提供五类能力：
1. 本地工具执行（shell / 文件 / MCP 工具）
2. Skill 加载与脚本执行
3. 把池内任何模型当"纯文本工人"调度（ask / ask_vote 等）
4. 完整的多智能体任务执行（run_task，本地选一个池内模型当主）
5. 持久会话（new_session / chat / list_sessions / close_session）——
   外部主可以开多个话茬，各自带历史上下文，也能续接上一次的对话。

协议：stdin 每行一个 JSON 请求，stdout 每行一个 JSON 响应，UTF-8。
流式（事件流）：请求带 "stream": true 时，中间过程逐行以
  {"event":"tool_start","name":...} / {"event":"stage","stage":...} 形式即时输出，
最后的正式结果仍是一行 {"ok":true,...} 响应。

请求指令：
  {"cmd":"ping"}
  {"cmd":"list_agents"}
  {"cmd":"list_tools"}
  {"cmd":"call_tool","name":"run_shell","args":{"command":"dir"}}
  {"cmd":"load_skill","name":"example_hello"}
  {"cmd":"run_skill_script","skill":"example_hello","script":"hello.py","args":{}}
  {"cmd":"ask","agent":"glm-flash","prompt":"...","system":"可选角色"}
  {"cmd":"ask_many","workers":["a","b"],"prompt":"..."}        （并行派多个工人）
  {"cmd":"ask_vote","prompt":"...","threshold":0.5}            （两步投票取共识）
  {"cmd":"run_task","task":"...","orchestrator":"可选","solo":false,"stream":true}
  {"cmd":"run_pipeline","task":"...","stream":true}            （固定流水线：起草→评审→修订）
  {"cmd":"new_session","session_id":"可选","orchestrator":"可选","tools":["白名单"]}  （建持久会话，可做工具隔离）
  {"cmd":"chat","message":"...","session_id":"...","stream":true} （用会话跑一轮，可续历史）
  {"cmd":"list_sessions"}
  {"cmd":"session_tools","session_id":"..."}                       （查某会话的工具白名单/被裁剪项）
  {"cmd":"cancel_tool","session_id":"..."}                         （结束当前工具派工；bridge 串行下 best-effort，Web 才实时）
  {"cmd":"close_session","session_id":"..."}

响应：{"ok":true,...} 或 {"ok":false,"error":"..."}；启动后先发一行 ready 事件。
"""
from __future__ import annotations

import json
import sys
import uuid
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core.agent import Agent

from . import __version__
from .config import Config
from .core.events import to_event
from .core.llm import LLMClient
from .core.orchestrator import WORKER_SYSTEM, WorkerPool
from .core.pipeline import Pipeline
from .core.session import sanitize_session_id
from .skills_manager import SkillManager, register_skill_tools
from .tools.base import ToolRegistry
from .tools.builtin import build_builtin_tools


def _norm_workers(raw) -> list[str] | None:
    """把请求里的 workers 字段规范成列表；缺省 None（用池内全部）。"""
    if not raw:
        return None
    if isinstance(raw, str):
        return [w.strip() for w in raw.split(",") if w.strip()] or None
    return [str(w) for w in raw if str(w).strip()] or None


def _norm_tools(raw) -> list[str] | None:
    """把请求里的 tools（工具白名单）规范成列表；缺省 None（放开全部）。"""
    if not raw:
        return None
    if isinstance(raw, str):
        return [w.strip() for w in raw.split(",") if w.strip()] or None
    return [str(w) for w in raw if str(w).strip()] or None


class Bridge:
    """一个 bridge 会话：共享的注册表 + 技能 + MCP + 工人池 + 持久会话表。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.skills = SkillManager()
        self.skills.scan()
        self.registry: ToolRegistry = build_builtin_tools(cfg)
        register_skill_tools(self.registry, self.skills)
        # MCP 走进程内共享连接组：bridge 与它临时建的 Agent（run_task）共享同一套
        # 连接，避免同进程内重复拉起 stdio 子进程、重复等最长 90s 的首连。
        self.mcp_group = None
        self.mcp_status: list[str] = []
        if cfg.mcp_servers:
            from .tools.mcp_client import acquire_group, register_group_tools
            self.mcp_group = acquire_group(cfg.mcp_servers)
            self.mcp_status = list(self.mcp_group.status)
            register_group_tools(self.registry, self.mcp_group)
        self.workers = WorkerPool(cfg)
        # 持久会话：外部主可开多个话茬，各带独立 Agent（含 JSONL 落盘的会话历史）
        # 用字符串延迟引用避免循环导入（Agent 在 _new_session 里局部引入）
        self._sessions: dict[str, Agent] = {}
        self._sessions_lock = Lock()
        # 兜底单模型客户端（ask 未指定 agent 时用）：首次访问创建后复用，
        # 避免每次 ask 都新建一个 OpenAI 客户端（各自一套 httpx 连接池，用完即弃）。
        # Bridge.handle 由单线程协议循环驱动，仍加锁以防将来多线程化。
        self._fallback_llm: LLMClient | None = None
        self._llm_lock = Lock()

    # ---------- 持久会话 ----------

    def _fallback_client(self) -> LLMClient:
        """返回兜底单模型客户端（懒创建 + 复用）。

        temperature/timeout/max_retries 与 Agent、WorkerPool 同源读 llm 配置：
        用代码默认值会让 config.yaml 的调整对这条路径不生效，
        免费节点友好策略在 bridge 下静默降级。
        """
        with self._llm_lock:
            if self._fallback_llm is None:
                llm = self.cfg.llm
                self._fallback_llm = LLMClient(
                    str(llm.get("base_url", "")),
                    str(llm.get("api_key", "")),
                    str(llm.get("model", "")),
                    temperature=self.cfg.as_float(
                        llm.get("temperature", 0.7), "llm.temperature", 0.7, minimum=0.0
                    ),
                    timeout=self.cfg.as_float(
                        llm.get("timeout", 120) or 120, "llm.timeout", 120.0, minimum=1.0
                    ),
                    max_retries=self.cfg.as_int(
                        llm.get("max_retries", 2) or 0, "llm.max_retries", 2, minimum=0
                    ),
                )
            return self._fallback_llm

    def _find_profile(self, orch: str | None):
        if orch:
            p = self.cfg.profile(orch)
            if p is None:
                raise ValueError(f"orchestrator '{orch}' 不在智能体池中")
            return p
        return self.cfg.agent_profiles[0] if self.cfg.agent_profiles else None

    def _new_session(self, session_id: str | None, orch: str | None = None, tools: list[str] | None = None) -> dict:
        from .core.agent import Agent

        sid = (session_id or "").strip()
        # 边界校验：session_id 会作为会话 JSONL 的文件名，非法字符/.. 一律拒绝
        # （而不是静默改写，否则调用方手里的 id 与落盘名会不一致）。
        if sid and sanitize_session_id(sid) is None:
            return {"ok": False, "error": "session_id 非法：只允许字母、数字、点、下划线、连字符，长度 1-64"}
        if not sid:
            sid = uuid.uuid4().hex[:12]
        # 先在锁外解析 profile：让"orchestrator 不在池中"等配置错误尽早暴露，
        # 而不是等 Agent 构造中途才炸（在锁内崩会阻塞其它并发会话请求）。
        profile = self._find_profile(orch)
        with self._sessions_lock:
            if sid in self._sessions:
                return {"ok": False, "error": f"会话 '{sid}' 已存在"}
            # Agent 构造（含 MCP 连接、技能扫描、可能慢）放锁内一次性完成，
            # 避免锁外构造时 sid 被并发覆盖。原行为：构造失败会抛异常，被
            # handle() 的 except 兜底返回错误，不会留下半成品。
            bot = Agent(
                self.cfg,
                session_id=sid,
                profile=profile,
                enable_workers=True,
                allow_tools=tools,
            )
            self._sessions[sid] = bot
        info = {"session_id": sid}
        if tools:
            info["allowed_tools"] = bot.registry.names()
        # MCP 连接失败告警：建会话时如实上报，不让故障静默
        if bot.mcp_status:
            info["mcp_warnings"] = bot.mcp_status
        return {"ok": True, **info}

    def _close_session(self, session_id: str) -> dict:
        with self._sessions_lock:
            bot = self._sessions.pop(session_id, None)
        if bot is None:
            return {"ok": False, "error": f"会话 '{session_id}' 不存在"}
        try:
            bot.close()
        except Exception:  # noqa: BLE001 - 关闭失败不影响结果
            pass
        return {"ok": True, "session_id": session_id}

    def _cancel_tool(self, req: dict) -> dict:
        """请求结束当前会话正在执行的工具（当前仅 ask_workers 会响应）。

        注意 bridge 的局限：bridge 是单线程串行读 stdin，`chat`/`run_task` 会阻塞整
        个循环直到本轮跑完，这条指令只能排在其后、事后送达，起不到「跑一半时打断」
        的作用。真正能用的是 Web 入口（轮次在后台线程跑，主循环可并发受理
        /api/chat/cancel_tool）。这里保留指令是为了协议对称与将来支持异步 chat。
        """
        sid = str(req.get("session_id", "")).strip()
        with self._sessions_lock:
            bot = self._sessions.get(sid)
        if bot is None:
            return {"ok": False, "error": f"会话 '{sid}' 不存在"}
        fn = getattr(bot, "cancel_current_tool", None)
        if fn is None:
            return {"ok": False, "error": "该会话不支持工具取消"}
        fn()
        return {"ok": True, "session_id": sid}

    # ---------- 指令分发 ----------

    def handle(self, req: dict, out=None) -> dict:
        """处理一条请求。out：可选的流式输出函数（写一行 JSON + flush）。

        返回最终响应 dict（serve 循环负责写 stdout）；stream 期间通过 out 边执行边推事件。
        """
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"ok": True, "version": __version__, "agents": [p.name for p in self.cfg.agent_profiles]}
        if cmd == "list_agents":
            return {"ok": True, "agents": [p.brief() for p in self.cfg.agent_profiles]}
        if cmd == "list_tools":
            return {"ok": True, "tools": self.registry.info_list()}
        if cmd == "call_tool":
            name = str(req.get("name", ""))
            return {"ok": True, "result": self.registry.execute(name, req.get("args") or {})}
        if cmd == "load_skill":
            return {"ok": True, "content": self.skills.load_full(str(req.get("name", "")))}
        if cmd == "run_skill_script":
            return {"ok": True, "result": self.skills.run_script(
                str(req.get("skill", "")), str(req.get("script", "")), req.get("args") or {}
            )}
        if cmd == "ask":
            prompt = str(req.get("prompt", "")).strip()
            if not prompt:
                return {"ok": False, "error": "prompt 不能为空"}
            agent = str(req.get("agent", "")).strip()
            if agent:
                return {"ok": True, "answer": self.workers.ask(agent, prompt, req.get("system"))}
            # 未指定 agent 时用兜底单模型（懒创建复用，配置读取见 _fallback_client）
            client = self._fallback_client()
            resp = client.chat([
                {"role": "system", "content": str(req.get("system") or WORKER_SYSTEM)},
                {"role": "user", "content": prompt},
            ])
            return {"ok": True, "answer": resp.get("content", "")}
        if cmd == "ask_many":
            prompt = str(req.get("prompt", "")).strip()
            if not prompt:
                return {"ok": False, "error": "prompt 不能为空"}
            return {"ok": True, "results": self.workers.ask_many(
                _norm_workers(req.get("workers")) or self.workers.pick(),
                prompt,
                req.get("system"),
            )}
        if cmd == "ask_vote":
            prompt = str(req.get("prompt", "")).strip()
            if not prompt:
                return {"ok": False, "error": "prompt 不能为空"}
            return {"ok": True, "result": self.workers.vote(prompt, _norm_workers(req.get("workers")), req.get("threshold"))}

        # ---- 持久会话 ----
        if cmd == "new_session":
            return self._new_session(
                req.get("session_id"), req.get("orchestrator"), _norm_tools(req.get("tools"))
            )
        if cmd == "list_sessions":
            with self._sessions_lock:
                return {"ok": True, "sessions": list(self._sessions)}
        if cmd == "session_tools":
            sid = str(req.get("session_id", "")).strip()
            with self._sessions_lock:
                bot = self._sessions.get(sid)
            if bot is None:
                return {"ok": False, "error": f"会话 '{sid}' 不存在"}
            removed = getattr(bot, "removed_tools", [])
            return {"ok": True, "session_id": sid,
                    "allowed_tools": bot.registry.names(), "removed_tools": removed}
        if cmd == "close_session":
            return self._close_session(str(req.get("session_id", "")))
        if cmd == "cancel_tool":
            return self._cancel_tool(req)
        if cmd == "chat":
            return self._chat(req, out)

        # ---- 一次性任务 ----
        if cmd == "run_task":
            return self._run_task(req, out)
        if cmd == "run_pipeline":
            return self._run_pipeline(req, out)
        return {"ok": False, "error": f"未知指令: {cmd}"}

    # ---------- 具体指令实现 ----------

    def _chat(self, req: dict, out) -> dict:
        """在持久会话里跑一轮对话。可续接该会话的历史（JSONL 落盘）。"""
        message = str(req.get("message", "")).strip()
        if not message:
            return {"ok": False, "error": "message 不能为空"}
        sid = str(req.get("session_id", "")).strip()
        if not sid:
            return {"ok": False, "error": "chat 需要 session_id，请先 new_session"}
        with self._sessions_lock:
            bot = self._sessions.get(sid)
        if bot is None:
            return {"ok": False, "error": f"会话 '{sid}' 不存在，请先 new_session 或改选其它会话"}

        def emit(ev: dict) -> None:
            if out:
                out(to_event(ev))

        try:
            result = bot.run(message, on_event=emit if req.get("stream") else None)
        except Exception as e:  # noqa: BLE001 - 单轮失败不搞挂会话
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return {"ok": True, "session_id": sid, "result": result}

    def _run_task(self, req: dict, out) -> dict:
        """完整跑一轮多智能体任务（临时 Agent，跑完即关）。支持 stream 事件流。"""
        task = str(req.get("task", "")).strip()
        if not task:
            return {"ok": False, "error": "task 不能为空"}
        solo = bool(req.get("solo"))
        orch = str(req.get("orchestrator") or "").strip()
        try:
            profile = self._find_profile(orch or None)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        from .core.agent import Agent

        bot = Agent(self.cfg, profile=profile, enable_workers=not solo)

        def emit(ev: dict) -> None:
            if out:
                out(to_event(ev))

        try:
            result = bot.run(task, on_event=emit if req.get("stream") else None)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            bot.close()
        return {"ok": True, "result": result}

    def _run_pipeline(self, req: dict, out) -> dict:
        """固定流水线：起草→评审→修订（不选主，直接调度池内全部模型）。"""
        task = str(req.get("task", "")).strip()
        if not task:
            return {"ok": False, "error": "task 不能为空"}
        if not self.workers.names():
            return {"ok": False, "error": "智能体池为空，无法执行流水线"}

        def emit(ev: dict) -> None:
            if out:
                out(to_event(ev))

        result = Pipeline(self.workers).run(
            task,
            draft_workers=_norm_workers(req.get("draft_workers")),
            review_workers=_norm_workers(req.get("review_workers")),
            revise_workers=_norm_workers(req.get("revise_workers")),
            on_event=emit if req.get("stream") else None,
        )
        return {"ok": True, "result": result}

    def close(self) -> None:
        """关闭全部资源：会话 Agent、共享工人池与兜底客户端。"""
        with self._sessions_lock:
            bots = list(self._sessions.values())
            self._sessions.clear()
        for b in bots:
            try:
                b.close()
            except Exception:  # noqa: BLE001
                pass
        # 共享工人池与兜底单模型客户端都持有 httpx 连接池，进程退出前显式释放。
        # MCP 只解除本使用者的引用（引用归零时组内部断开）。
        for holder in (self.workers, self._fallback_llm):
            if holder is None:
                continue
            try:
                holder.close()
            except Exception:  # noqa: BLE001
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
    由 main()（bridge 子命令、load_config 之前）与 serve() 双入口调用，
    保证警告在配置加载阶段打印时也已是 UTF-8 字节。
    """
    for stream in (sys.stdout, sys.stdin, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001 - 老版本 Python 或非 tty 场景尽力而为
            pass


def serve(cfg: Config) -> None:
    """阻塞式协议循环：stdin 请求行 → stdout 响应行。"""
    reconfigure_streams()
    # 启动时清一次历史会话（只留最近 30 个），防 data/sessions 无限膨胀
    from .core.session import Session
    Session.cleanup()
    bridge = Bridge(cfg)
    ready = {
        "ok": True,
        "event": "ready",
        "version": __version__,
        "agents": [p.name for p in bridge.cfg.agent_profiles],
        "mcp": bridge.mcp_status,
    }
    sys.stdout.write(json.dumps(ready, ensure_ascii=False) + "\n")
    sys.stdout.flush()

    def write_line(obj: dict) -> None:
        """写出任意行：事件行或最终响应行。"""
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                if isinstance(req, dict):
                    resp = bridge.handle(req, out=write_line)
                else:
                    resp = {"ok": False, "error": "请求必须是 JSON 对象"}
            except json.JSONDecodeError as e:
                resp = {"ok": False, "error": f"JSON 解析失败: {e}"}
            except Exception as e:  # noqa: BLE001 - 单条指令失败不能拖垮协议循环
                resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            write_line(resp)
    finally:
        bridge.close()
