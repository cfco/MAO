"""主控 Agent Loop：调模型 → 解析工具调用 → 执行 → 结果回灌 → 循环。

v2 协作框架：
- 主智能体（本类）：拆解、派工、汇总、把关，掌握全部本地工具
  （内置 shell/文件 + MCP + Skill + ask_worker/ask_workers 派工工具）。
- 工人池（WorkerPool）：池里除主之外的其它模型，纯文本执行器。
  由 profile / enable_workers 控制；主未启用协作或池里只有主时退化为 solo。
"""
from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable

from ..config import DEFAULT_CONTEXT, ROOT, AgentProfile, Config
from ..skills_manager import SkillManager, register_skill_tools
from ..tools.base import ToolRegistry
from ..tools.builtin import build_builtin_tools
from .llm import LLMClient, LLMError
from .orchestrator import WorkerPool, register_worker_tools
from .pipeline import Pipeline, register_pipeline_tool
from .session import Session

SYSTEM_PROMPT_TEMPLATE = """你是运行在用户 Windows 电脑上的多功能智能体。

当前日期：{today}
项目根目录：{root}（相对路径都基于这里）

工作原则：
- 涉及时效信息、不确定的事实，先用工具查证，不要凭记忆编造。
- 多步任务按步骤推进，关键结果主动验证。
- 删除文件、覆盖数据等不可逆操作，先向用户说明影响并等确认。
- 回答用中文，简洁直接。

# 可用技能
{skills}

当某个技能与当前任务相关时，先调用 load_skill 读取技能全文再行动；技能脚本可用 execute_skill_script 执行。

# 协作工人池
{workers}
{workers_guide}"""

WORKERS_GUIDE = """你不是一个人在工作：工人池里的模型可以帮你干活，善用它们是质量的保证。
- 复杂任务先拆解成相互独立的子任务，用 ask_worker 派出去并行干，你负责汇总和把关。
- 影响最终质量的关键问题（方案选型、结论判断、代码审查），用 ask_workers 同时派多人，对比回答取共识或择优。
- 工人的结果必须经过你的审查才能采信，不能不加验证照搬；发现错误要指出并要求重做或自己修正。
- 简单小事自己直接做完即可，不必事事派工。"""

EventCallback = Callable[[dict], None]

# 单次工具结果塞进上下文的最大字节数（防 MCP/Skill/ask_worker 长结果撑爆窗口）
MAX_TOOL_RESULT_BYTES = 32 * 1024


class Agent:
    def __init__(
        self,
        cfg: Config,
        session_id: str | None = None,
        profile: AgentProfile | None = None,
        enable_workers: bool = True,
        allow_tools: list[str] | None = None,
    ):
        """构造主智能体。

        cfg            全局配置
        session_id     会话 id（None 则自动生成）
        profile        主智能体配置；None 时退回 cfg.llm 兜底单模型
        enable_workers 是否启用协作工人池（solo 模式下为 False）
        allow_tools    工具白名单（会话级隔离）；None 表示放开全部工具。
                       注意：白名单会裁剪 worker/pipeline 协作工具，
                       需要协作就把 ask_worker 等一并加入白名单。
        """
        self.cfg = cfg
        self.profile = profile
        self.allow_tools = allow_tools

        # 主智能体 LLM：优先用选中的池内 profile；没有则用兜底单模型（v1 兼容）
        base_url = profile.base_url if profile else str(cfg.llm.get("base_url", ""))
        api_key = profile.api_key if profile else str(cfg.llm.get("api_key", ""))
        model = profile.model if profile else str(cfg.llm.get("model", ""))
        self.llm = LLMClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=float(cfg.llm.get("temperature", 0.7)),
            timeout=float(cfg.llm.get("timeout", 120)),
            max_retries=int(cfg.llm.get("max_retries", 2)),
        )

        # 技能 + 内置工具（任何配置下都加载）
        self.skills = SkillManager()
        self.skills.scan()
        self.registry: ToolRegistry = build_builtin_tools(cfg)
        register_skill_tools(self.registry, self.skills)

        # MCP 连接（可选，运行时失败不阻塞启动）
        self.mcp_connections: list = []
        self.mcp_status: list[str] = []
        if cfg.mcp_servers:
            from ..tools.mcp_client import connect_and_register

            self.mcp_status = connect_and_register(
                cfg.mcp_servers, self.registry, self.mcp_connections
            )

        # 协作工人池：主启用协作、且池里存在除主之外的模型时才创建。
        # 创建后把 ask_worker / ask_workers 注册为主智能体的派工工具。
        self.worker_pool: WorkerPool | None = None
        if enable_workers:
            pool = WorkerPool(cfg, exclude=profile.name if profile else None)
            if pool.names():
                self.worker_pool = pool
                register_worker_tools(self.registry, pool)
                # 固定流水线工具：run_pipeline（起草→评审→修订）
                register_pipeline_tool(self.registry, lambda: Pipeline(pool))

        # 会话级工具隔离：白名单裁剪（在全部工具注册完成后执行）
        # 无条件初始化默认值，去掉外部依赖 getattr(bot, "removed_tools", []) 兜底
        self.removed_tools: list[str] = []
        # 显式校验：None = 放开全部；非 None（包括空列表）= 白名单
        # 空列表容易被误以为是"不限制"，实际是"全部禁用" → 模型一个工具也调不到。
        # 在这里显式警告提示，避免用户误用导致 Agent 一上来就死循环。
        if allow_tools is not None and not allow_tools:
            # 走 stderr：bridge 协议与 --stream 的 stdout 是纯 JSON 行，不能被提示混入
            print("[警告] allow_tools=[] 表示禁用全部工具，Agent 将无法调用任何工具。"
                  "如需放开全部工具请传 None，传非空列表表示白名单。", file=sys.stderr)
        # 用 `is not None` 判定而非 `if allow_tools`：保证空列表也走 restrict 路径
        if allow_tools is not None:
            removed = self.registry.restrict(allow_tools)
            if removed:
                self.removed_tools = removed

        self.session = Session(
            session_id,
            flush_batch=cfg.session_flush_batch,
            flush_interval=cfg.session_flush_interval,
        )
        # 会话级并发闸门：Web 允许同一 session_id 并发提交（两个 SSE 请求
        # 各起一个线程跑同一个 Agent），并发跑会交错写 session.history 与
        # 落盘缓冲，会话记录错乱。run() 用非阻塞抢锁快速失败，不排队。
        self._run_lock = threading.Lock()

    def close(self) -> None:
        """释放资源：刷盘会话缓冲，关闭 MCP 长连接，并释放 LLM 连接池。调用方（CLI/Web/桥）结束时应调用。"""
        session = getattr(self, "session", None)
        if session is not None:
            try:
                session.flush()  # 把未落盘的会话消息写盘，防正常退出时丢历史
            except Exception:  # noqa: BLE001 - 刷盘失败不应影响资源释放
                pass
        if self.mcp_connections:
            from ..tools.mcp_client import stop_all

            stop_all(self.mcp_connections)
            self.mcp_connections = []
        # 释放 HTTP 连接池：主模型 + 本 Agent 自有的工人池都持有 httpx 连接池，
        # 不显式关就只能等 GC 回收——Web 会话频繁创建/淘汰、bridge 常驻时会积压废弃连接。
        # WorkerPool 是每个 Agent 各建一个（见 __init__），所以这里关掉不会影响别的会话。
        for holder in (self.llm, self.worker_pool):
            if holder is None:
                continue
            try:
                holder.close()
            except Exception:  # noqa: BLE001 - 释放失败不应影响收尾
                pass

    def system_prompt(self) -> str:
        if self.worker_pool is not None:
            workers = self.worker_pool.overview()
            guide = WORKERS_GUIDE
        else:
            workers = "（未启用协作工人池，所有工作由你自己完成）"
            guide = ""
        return SYSTEM_PROMPT_TEMPLATE.format(
            today=time.strftime("%Y-%m-%d %A"),
            root=str(ROOT),
            skills=self.skills.overview(),
            workers=workers,
            workers_guide=guide,
        )

    def run(
        self,
        user_input: str,
        on_event: EventCallback | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        """处理一轮用户输入，直到模型给出最终回复。

        同一 Agent（=同一会话）同一时刻只允许一轮在跑：抢不到锁立即返回
        "会话忙"提示而不是排队——排队会让前端等待时间叠加、语义更差。
        busy 轮不写入会话历史（本轮实际什么都没发生）。

        on_event: 事件回调，供 CLI/Web 展示过程。事件类型：
          llm_call / tool_start{name} / tool_result{name,result} / final{content}
        should_stop: 可选的协作式取消探针，返回 True 时本轮在下一个检查点停下。
          检查点：每轮迭代开头（即每次 LLM 调用前）与每个工具执行前。
          注意粒度边界：已经在途的 LLM HTTP 请求无法同步中断，只能等它自然
          结束（由 cfg.llm.timeout 兜底上限）；进行中的工具调用同理不可打断，
          所以取消最坏延迟 ≈ 一次请求/工具调用的耗时。
        """

        def emit(event: dict) -> None:
            if on_event:
                on_event(event)

        if not self._run_lock.acquire(blocking=False):
            busy = "（会话忙：上一轮任务仍在执行中，本次输入未处理，请稍候再试。）"
            emit({"type": "final", "content": busy})
            return busy
        try:
            return self._run_turn(user_input, emit, should_stop)
        finally:
            self._run_lock.release()

    def _run_turn(
        self,
        user_input: str,
        emit: EventCallback,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        # 会话写入失败（磁盘满/权限等）不应让整轮崩溃，给出可理解的错误说明。
        try:
            self.session.add("user", user_input)
        except Exception as e:  # noqa: BLE001 - 会话落盘/内存异常不应搞崩整轮
            msg_err = f"会话写入失败：{type(e).__name__}: {e}"
            emit({"type": "error", "message": msg_err})
            emit({"type": "final", "content": msg_err})
            return msg_err
        ctx = (self.profile.context_length if self.profile else 0) or DEFAULT_CONTEXT
        messages = self.session.messages_with(self.system_prompt(), context_length=ctx)
        schemas = self.registry.openai_schemas()

        for _ in range(self.cfg.max_iterations):
            if should_stop is not None and should_stop():
                return self._interrupted(emit)
            emit({"type": "llm_call"})
            try:
                msg = self.llm.chat(messages, tools=schemas or None)
            except LLMError as e:
                # 免费节点友好：主节点失败不崩整体，给出可行动的错误说明
                msg_err = (f"模型调用失败[{e.error_type}]：{e}"
                           + ("（提示：稍后可重试；也可换一个可用节点当主或让工人代跑）" if e.retryable else ""))
                self.session.add("assistant", msg_err)
                emit({"type": "error", "message": msg_err})
                emit({"type": "final", "content": msg_err})
                return msg_err
            except Exception as e:  # noqa: BLE001 - 非 LLMError 的意外异常（网络库裸异常等）也不该崩整轮
                msg_err = f"模型调用异常：{type(e).__name__}: {e}"
                self.session.add("assistant", msg_err)
                emit({"type": "error", "message": msg_err})
                emit({"type": "final", "content": msg_err})
                return msg_err
            messages.append(msg)

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                content = (msg.get("content") or "").strip()
                if not content:
                    # 部分节点在内容过滤或"只吐思维链"时给出空 content 且无工具调用。
                    # 原样返回空串会让用户收到一条空白回复（且已写进会话历史），显式说明更好。
                    content = ("（模型返回了空内容：可能是节点内容过滤或只输出思维链，"
                               "可重试，或换一个可用节点当主。）")
                self.session.add("assistant", content)
                emit({"type": "final", "content": content})
                return content

            for tc in tool_calls:
                # 取消检查在每个工具执行前：本轮已发出的工具调用不回溯打断，
                # 但不再继续执行剩余工具，避免断开后把整串副作用跑完。
                if should_stop is not None and should_stop():
                    return self._interrupted(emit)
                name = tc["function"]["name"]
                emit({"type": "tool_start", "name": name})
                result = self.registry.execute(name, tc["function"]["arguments"])
                emit({"type": "tool_result", "name": name, "result": result})
                # 工具结果截断：超长只留前 MAX_TOOL_RESULT_BYTES 字节，防撑爆上下文窗口。
                # 注意必须按字节截断再解码：中文 UTF-8 占 3B/字，若按字符切片，
                # 32K 字符的中文结果实际仍可达 ~96KB，截断形同虚设。
                raw_result = result.encode("utf-8")
                if len(raw_result) > MAX_TOOL_RESULT_BYTES:
                    result = raw_result[:MAX_TOOL_RESULT_BYTES].decode("utf-8", errors="ignore") \
                        + f"\n...[已截断，超过 {MAX_TOOL_RESULT_BYTES}B]"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        fallback = "（已达单次任务最大循环轮数，先停在这里。你可以让我继续。）"
        self.session.add("assistant", fallback)
        emit({"type": "final", "content": fallback})
        return fallback

    def _interrupted(self, emit: EventCallback) -> str:
        """协作式取消的统一收尾：写入会话历史再发 final，
        让下一轮上下文知道上一轮是被取消的、并非已完成。"""
        note = "（本轮已被取消：客户端断开或用户中止，后续步骤未执行。）"
        self.session.add("assistant", note)
        emit({"type": "final", "content": note})
        return note
