"""Web 服务：POST /api/chat（SSE 流式事件）+ 单页对话界面。

启动：python web/server.py   或   uvicorn web.server:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import queue
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import Config, load_config  # noqa: E402
from agent.core.agent import Agent  # noqa: E402
from agent.core.events import error_event, new_session_event, to_event  # noqa: E402
from agent.core.session import sanitize_session_id  # noqa: E402

# 配置加载直接交给 load_config() —— 它内部已带 mtime 缓存：
# - mtime 未变 → 命中缓存，不重读磁盘
# - mtime 变化 → 自动重新解析 yaml，构建新 Config 对象
# web 层不再维护第二层 mtime 缓存，避免每次请求都做两次 stat 比较。
# 已有 Agent 持有的旧 Config 引用不受影响（避免断 MCP 长连接）。


def _get_cfg() -> Config:
    """获取当前最新 Config：load_config() 内部已处理 mtime 热加载。"""
    return load_config()


# 会话表：session_id -> Agent，外加每个会话绑定的主智能体名。
# 记主智能体名是为了识别"顶栏切了主"：Agent 在创建时就固定了主模型，
# 复用同一会话时新的 orchestrator 参数会被忽略（切换静默失效）。
# 后台线程定期清理超时未活跃的会话，避免 Agent/MCP 连接泄漏。
_agents: dict[str, Agent] = {}
_last_active: dict[str, float] = {}
_session_orch: dict[str, str] = {}
_lock = threading.Lock()
# 清理线程停止信号（lifespan 关闭时置位）
_stop_event = threading.Event()
IDLE_TIMEOUT = 30 * 60  # 会话空闲 30 分钟清理
CLEAN_INTERVAL = 60     # 清理线程扫描间隔
MAX_SESSIONS = 50       # 会话表硬上限，超限淘汰最久未活跃的


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    orchestrator: str | None = None


def _cleanup_loop() -> None:
    """后台清理线程：定期关闭并移除超时空闲会话。"""
    while not _stop_event.is_set():
        _stop_event.wait(CLEAN_INTERVAL)
        # 先找出要清理的会话列表（拷贝副本，避免边迭代边修改）
        with _lock:
            now = time.time()
            stale_ids = [
                sid for sid, t in _last_active.items()
                if now - t > IDLE_TIMEOUT
            ]
            # 把 Agent 引用移出字典，准备 close（close 可能较慢，放锁外执行）
            stale_agents = []
            for sid in stale_ids:
                agent = _agents.pop(sid, None)
                _last_active.pop(sid, None)
                _session_orch.pop(sid, None)
                if agent:
                    stale_agents.append(agent)
        # 锁外 close，避免阻塞其它并发请求；close 失败不影响（下次 scan 自然消失）
        for agent in stale_agents:
            try:
                agent.close()
            except Exception:  # noqa: BLE001 - 清理失败不阻塞其它会话
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期（替代已废弃的 @app.on_event）。

    启动：清理历史会话 → 预热 Agent 并打印 MCP 状态 → 拉起会话清理线程。
    关闭：停止清理线程 → 释放全部 Agent 资源。
    """
    cfg = _get_cfg()
    # 启动时清一次历史会话（只留最近 30 个），防 data/sessions 无限膨胀
    from agent.core.session import Session
    removed = Session.cleanup()
    if removed:
        print(f"[清理] 已清理 {removed} 个过期会话文件")
    print("启动 Web 服务前先做一次 Agent 初始化（检查模型配置与 MCP 连接）：")
    bot = Agent(cfg)
    for line in bot.mcp_status:
        print(line)
    bot.close()  # 测试用 Agent 也释放连接，不占用后台线程
    threading.Thread(target=_cleanup_loop, name="session-cleanup", daemon=True).start()
    try:
        yield
    finally:
        _stop_event.set()
        with _lock:
            agents = list(_agents.values())
            _agents.clear()
            _last_active.clear()
        for a in agents:
            try:
                a.close()
            except Exception:  # noqa: BLE001
                pass


app = FastAPI(title="自研多功能智能体", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/reload")
def reload_config():
    """手动触发热加载：强制重读 config.yaml。返回新旧配置的关键差异摘要。

    注意：已有 Agent 持有的旧 Config 引用不会被替换（避免断 MCP 长连接）；
    新创建的 Agent 会立即使用新配置。
    """
    # 取当前缓存的 Config 作为 old 做差异对比，再 force 重读得到 new
    old_cfg = load_config()  # 命中缓存，不会触发 reload
    new_cfg = load_config(force=True)  # 强制重读并重建

    # 简易差异：池内智能体数量变化 + llm timeout/max_retries 变化
    diff: list[str] = []
    if old_cfg is not new_cfg:
        diff.append(f"池内智能体：{len(old_cfg.agent_profiles)} → {len(new_cfg.agent_profiles)}")
        if old_cfg.llm_cfg.get("timeout") != new_cfg.llm_cfg.get("timeout"):
            diff.append(f"llm.timeout: {old_cfg.llm_cfg.get('timeout')} → {new_cfg.llm_cfg.get('timeout')}")
        if old_cfg.llm_cfg.get("max_retries") != new_cfg.llm_cfg.get("max_retries"):
            diff.append(f"llm.max_retries: {old_cfg.llm_cfg.get('max_retries')} → {new_cfg.llm_cfg.get('max_retries')}")
        old_names = {p.name for p in old_cfg.agent_profiles}
        new_names = {p.name for p in new_cfg.agent_profiles}
        added = new_names - old_names
        removed = old_names - new_names
        if added:
            diff.append(f"新增智能体：{', '.join(added)}")
        if removed:
            diff.append(f"移除智能体：{', '.join(removed)}")
    else:
        diff.append("配置无变化（或已是最新）")

    return {"ok": True, "changes": diff}


@app.get("/api/agents")
def list_agents():
    """智能体池清单（供前端选择主智能体）。"""
    cfg = _get_cfg()
    return {"agents": [p.brief() for p in cfg.agent_profiles]}


@app.post("/api/chat")
def chat(req: ChatRequest):
    message = (req.message or "").strip()
    if not message:
        return {"error": "消息不能为空"}

    # session_id 会作为会话 JSONL 的文件名：非法字符/.. 直接报错（不静默改写，
    # 否则前端手里的 id 与落盘名不一致，下次续接会话会失效）。
    sid = (req.session_id or "").strip()
    if sid:
        safe_sid = sanitize_session_id(sid)
        if safe_sid is None:
            return {"error": "session_id 非法：只允许字母、数字、点、下划线、连字符，长度 1-64"}
        sid = safe_sid
    else:
        sid = uuid.uuid4().hex[:12]

    # 先把"本次请求想要的主智能体"解析出来（配置在锁外读，避免锁内做配置解析）
    cfg = _get_cfg()
    profile = cfg.profile(req.orchestrator) if req.orchestrator else None
    if profile is None and cfg.agent_profiles:
        profile = cfg.agent_profiles[0]
    want_orch = profile.name if profile else ""

    evicted_agents: list[Agent] = []  # 被淘汰/被换主的旧会话，锁外关闭
    need_build = False
    with _lock:
        agent = _agents.get(sid)
        # 同一 session_id 换了主智能体 → 旧 Agent 绑的是旧主，复用会让"顶栏切换主"
        # 静默失效。按"切换即新会话"语义丢弃旧会话，用新主重建。
        if agent is not None and _session_orch.get(sid, "") != want_orch:
            _agents.pop(sid, None)
            _session_orch.pop(sid, None)
            evicted_agents.append(agent)
            agent = None
        if agent is None:
            # 硬上限：超限时先淘汰最久未活跃的会话（close 放锁外执行）
            excess = len(_agents) - MAX_SESSIONS + 1
            if excess > 0:
                oldest = sorted(_last_active.items(), key=lambda kv: kv[1])[:excess]
                for old_sid, _ in oldest:
                    old_agent = _agents.pop(old_sid, None)
                    _last_active.pop(old_sid, None)
                    _session_orch.pop(old_sid, None)
                    if old_agent:
                        evicted_agents.append(old_agent)
            # 先占位活跃时间，让并发的同会话请求不重复构造 Agent
            _last_active[sid] = time.time()
            need_build = True
        else:
            _last_active[sid] = time.time()

    if need_build:
        # Agent 构造会同步等待 MCP 连接（每个 server 最长 90s），必须在锁外执行：
        # 原实现在全局锁内构造，单个新会话就能把其它请求与清理线程全部堵住。
        try:
            built = Agent(cfg, session_id=sid, profile=profile, enable_workers=True)
        except Exception as e:  # noqa: BLE001 - 构造失败要回明确的 HTTP 错误而非 500
            with _lock:
                # 只清自己留下的占位；并发分支可能已成功建好同一会话，别把它的活跃时间抹掉
                if sid not in _agents:
                    _last_active.pop(sid, None)
            return {"error": f"会话初始化失败：{type(e).__name__}: {e}"}
        with _lock:
            winner = _agents.get(sid)
            if winner is None:
                _agents[sid] = built
                _session_orch[sid] = want_orch
                winner, built = built, None
        if built is not None:
            # 并发下其它请求已建好同一会话：丢弃自己这份，避免 MCP 连接重复占用
            try:
                built.close()
            except Exception:  # noqa: BLE001
                pass
        agent = winner

    # 锁外关闭被淘汰/换主的旧会话（close 可能较慢，放锁外不阻塞新会话创建）
    for a in evicted_agents:
        try:
            a.close()
        except Exception:  # noqa: BLE001 - 关闭失败静默忽略，下次清理会自然消失
            pass

    events: queue.Queue = queue.Queue()
    # 客户端断开 → 取消后台轮次：生成器 finally 置位 cancel_event，
    # 后台 Agent.run 经 should_stop 在迭代/工具边界协作式停下，停止消耗 token。
    cancel_event = threading.Event()

    def on_event(ev: dict) -> None:
        events.put(ev)

    def run() -> None:
        try:
            events.put(new_session_event(sid))
            agent.run(message, on_event=on_event, should_stop=cancel_event.is_set)
        except Exception as e:  # noqa: BLE001
            events.put(error_event(f"{type(e).__name__}: {e}"))
        finally:
            events.put(None)

    threading.Thread(target=run, daemon=True).start()

    async def gen():
        # 为什么是 async 生成器而非原来的同步生成器：同步生成器经 threadpool
        # 驱动，断连时正阻塞在 events.get() 里，Starlette 无法及时把取消传进去，
        # 取消语义形同虚设。async 生成器在客户端断开后、下一个 await 点必定收到
        # GeneratorExit/CancelledError，finally 得以执行置位。
        # 轮询 get_nowait + 50ms 休眠：本生成器感知断开 ≈50ms；后台轮次的取消
        # 粒度还叠加一次在途 LLM 请求/工具调用的耗时（无法同步打断，见 Agent.run）。
        try:
            while True:
                try:
                    ev = events.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.05)
                    continue
                if ev is None:
                    yield "data: [DONE]\n\n"
                    break
                yield f"data: {json.dumps(to_event(ev) if ev.get('type') else ev, ensure_ascii=False)}\n\n"
        finally:
            # 正常结束（[DONE]）与断开关闭都走这里；对已结束的轮次置位是无害幂等
            cancel_event.set()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main() -> None:
    import uvicorn

    # 预热、历史会话清理、清理线程启动、关闭资源释放都由 app 的 lifespan 统一处理
    cfg = _get_cfg()
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
