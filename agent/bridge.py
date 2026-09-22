"""外部智能体驱动模式（bridge）：谁启动并驱动本项目，谁就是主智能体。

本项目为驱动者（外部主 AI）提供三类能力，围绕一个目标——**用多个免费 key 帮主 AI
干脏活、并行/交叉验证，省主 AI 的 token**：
1. 本地工具执行（shell / 文件 / MCP 工具 / Skill）——给被沙箱限制的主一双"手"。
2. 按需调用池内免费模型当"纯文本子 AI"：ask（单个）/ ask_many（并行+结构化）/
   ask_vote（两步投票取共识）/ run_review（主给初稿、子 AI 只当评审团挑错）。
3. 派工前预检：health（各子 AI 冷却/当日隔离/能力标签）、list_agents（池清单）。

协议：stdin 每行一个 JSON 请求，stdout 每行一个 JSON 响应，UTF-8；启动即发一行 ready。
诊断/警告一律走 stderr，stdout 只放 JSON 行，供外部主按行 json.loads 解析。

请求指令：
  {"cmd":"ping"}
  {"cmd":"list_agents"}                                   （池清单，含能力标签，不含 key）
  {"cmd":"health"}                                        （各子 AI 可用性快照：冷却/当日隔离/标签）
  {"cmd":"list_tools"}
  {"cmd":"call_tool","name":"run_shell","args":{"command":"dir"}}
  {"cmd":"load_skill","name":"example_hello"}
  {"cmd":"run_skill_script","skill":"example_hello","script":"hello.py","args":{}}
  {"cmd":"ask","agent":"glm-flash","prompt":"...","system":"可选角色"}
  {"cmd":"ask_many","workers":["a","b"],"prompt":"..."}    （并行派工，返回 workers 结构化 + results 文本）
  {"cmd":"ask_vote","prompt":"...","threshold":0.5}         （两步投票取共识）
  {"cmd":"run_review","draft":"...","context":"可选","workers":["a","b"]}  （外部主给初稿，子 AI 只当评审团）

响应：{"ok":true,...} 或 {"ok":false,"error":"..."}。
"""
from __future__ import annotations

import json
import sys
from threading import Lock

from . import __version__
from .config import Config
from .core.llm import LLMClient
from .core.orchestrator import WORKER_SYSTEM, WorkerPool
from .skills_manager import SkillManager, register_skill_tools
from .tools.base import ToolRegistry
from .tools.builtin import build_builtin_tools

REVIEW_SYSTEM = (
    "你是独立评审智能体。针对给你的稿件只挑问题、给改进方向与编号建议，"
    "不改写全文、不泛泛夸赞。若整体可用就明确说'可用'并列出仅剩的小问题。"
)


def _norm_workers(raw) -> list[str] | None:
    """把请求里的 workers 字段规范成列表；缺省 None（用 pick() 的可用工人）。"""
    if not raw:
        return None
    if isinstance(raw, str):
        return [w.strip() for w in raw.split(",") if w.strip()] or None
    return [str(w) for w in raw if str(w).strip()] or None


class Bridge:
    """一个 bridge 连接：共享的注册表 + 技能 + MCP 连接组 + 工人池 + 兜底单模型。"""

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
            self.mcp_group = acquire_group(cfg.mcp_servers)
            self.mcp_status = list(self.mcp_group.status)
            register_group_tools(self.registry, self.mcp_group)
        self.workers = WorkerPool(cfg)
        # 兜底单模型客户端（ask 未指定 agent 时用）：首次访问创建后复用，
        # 避免每次 ask 都新建一个 OpenAI 客户端（各自一套 httpx 连接池，用完即弃）。
        self._fallback_llm: LLMClient | None = None
        self._llm_lock = Lock()

    # ---------- 兜底单模型 ----------

    def _fallback_client(self) -> LLMClient:
        """返回兜底单模型客户端（懒创建 + 复用）。

        temperature/timeout/max_retries 与 WorkerPool 同源读 llm 配置：用代码默认值
        会让 config.yaml 的调整对这条路径不生效，免费节点友好策略在 bridge 下静默降级。
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

    # ---------- 指令分发 ----------

    def handle(self, req: dict) -> dict:
        """处理一条请求，返回最终响应 dict（serve 循环负责写 stdout）。"""
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"ok": True, "version": __version__, "agents": [p.name for p in self.cfg.agent_profiles]}
        if cmd == "list_agents":
            return {"ok": True, "agents": [p.brief() for p in self.cfg.agent_profiles]}
        if cmd == "list_tools":
            return {"ok": True, "tools": self.registry.info_list()}
        if cmd == "health":
            # 每个子 AI 的当前可用性快照（冷却/当日隔离/能力标签），供外部主派工前预检
            return {"ok": True, "workers": self.workers.health_status()}
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
            # 一次收集，同时回结构化（外部主程序化消费）与文本（人读/兼容）
            recs = self.workers.ask_many_structured(
                _norm_workers(req.get("workers")) or self.workers.pick(),
                prompt,
                req.get("system"),
            )
            text = "\n\n".join(
                f"### 工人 {r['worker']} 的结果\n" + (r["answer"] if r["ok"] else f"（{r['status']}）")
                for r in recs
            )
            return {"ok": True, "results": text, "workers": recs}
        if cmd == "ask_vote":
            prompt = str(req.get("prompt", "")).strip()
            if not prompt:
                return {"ok": False, "error": "prompt 不能为空"}
            return {"ok": True, "result": self.workers.vote(prompt, _norm_workers(req.get("workers")), req.get("threshold"))}
        if cmd == "run_review":
            draft = str(req.get("draft", "")).strip()
            if not draft:
                return {"ok": False, "error": "draft 不能为空（外部主给出初稿，子 AI 只当评审团）"}
            context = str(req.get("context", "")).strip()
            reviewers = _norm_workers(req.get("workers")) or self.workers.pick()
            prompt = (
                "请作为独立评审，针对下面的稿件给出问题清单与**编号**改进建议，"
                "直接挑错、给方向，不要客套、不要整段复述原文。\n"
                + (f"【背景 / 验收标准】{context}\n" if context else "")
                + f"【待评审稿件】\n{draft}"
            )
            recs = self.workers.ask_many_structured(reviewers, prompt, REVIEW_SYSTEM)
            return {"ok": True, "reviews": recs}
        return {"ok": False, "error": f"未知指令: {cmd}"}

    def close(self) -> None:
        """释放共享工人池、兜底单模型客户端与 MCP 连接组引用。"""
        # 工人池与兜底客户端都持有 httpx 连接池，进程退出前显式释放。
        # MCP 只解除本使用者的引用（引用归零时组内部才真正断开）。
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
    由 main()（bridge 子命令、load_config 之前）与 serve() 双入口调用。
    """
    for stream in (sys.stdout, sys.stdin, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001 - 老版本 Python 或非 tty 场景尽力而为
            pass


def serve(cfg: Config) -> None:
    """阻塞式协议循环：stdin 请求行 → stdout 响应行。"""
    reconfigure_streams()
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
                resp = {"ok": False, "error": f"JSON 解析失败: {e}"}
            except Exception as e:  # noqa: BLE001 - 单条指令失败不能拖垮协议循环
                resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    finally:
        bridge.close()
