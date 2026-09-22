"""对外事件统一 schema：CLI(--stream) / bridge / Web(SSE) 三个入口共用。

约定（所有入口一致，脚本/前端只解析这一种格式）：
  {"event":"llm_call"}
  {"event":"tool_start","name":...}
  {"event":"tool_result","name":...,"preview":"截断后的结果(≤500字)"}
  {"event":"stage","stage":"draft|review|revise","summary":...}
  {"event":"final","content":"完整最终答案（不截断）"}
  {"event":"error","message":...}          # 单轮内出错
  {"event":"session","session_id":...}     # Web 建会话通知；协议层用 ok+id 响应
  # —— ask_workers 并行派工的逐工人实时进度（快工人先回来，用户不必等最慢的）——
  {"event":"worker_dispatch","workers":[...],"total":N,"prompt_preview":"..."}
  {"event":"worker_result","worker":...,"index":i,"total":N,"elapsed_ms":...,"ok":true|false,"preview":"..."}
  {"event":"worker_gather_cancelled","completed":[...],"skipped":[...]}

worker_result 的 `ok` 区分"成功带回答案"与"失败/超时/当日隔离"，前端据此把骨架格
标对勾或叉。`worker_gather_cancelled` 只在用户点「采纳已完成、结束派工」后出现，
标记哪些工人被放弃等待（其 LLM 请求已在途，不追溯撤单，后台自行收尾）。

从 Agent / Pipeline 内部事件回调（{"type": ...}）转换到统一对外格式。
长文本只在中间过程截断（tool_result / worker_result / stage），最终答案(final)不截断。
"""
from __future__ import annotations

TOOL_PREVIEW_LEN = 500
STAGE_SUMMARY_LEN = 200


def to_event(ev: dict) -> dict:
    """把 Agent/Pipeline 的事件回调 dict 转成统一对外事件行。

    各 type 的映射都显式列出：未识别的 type 兜底透传（保留原有 type 转 event）。
    error 类型单独显式归一（即便当前透传逻辑也能跑通，显式列出让 schema 一目了然）。
    """
    t = ev.get("type")
    if t == "tool_start":
        return {"event": "tool_start", "name": ev.get("name")}
    if t == "tool_result":
        return {
            "event": "tool_result",
            "name": ev.get("name"),
            "preview": str(ev.get("result", ""))[:TOOL_PREVIEW_LEN],
        }
    if t == "llm_call":
        return {"event": "llm_call"}
    if t == "pipeline_stage":
        return {
            "event": "stage",
            "stage": ev.get("stage"),
            "summary": str(ev.get("summary", ""))[:STAGE_SUMMARY_LEN],
        }
    if t == "final":
        return {"event": "final", "content": str(ev.get("content", ""))}
    if t == "error":
        # 显式归一：将来若 Agent 内部 error 事件结构变化，这里集中处理
        return {"event": "error", "message": str(ev.get("message", ""))}
    if t == "worker_dispatch":
        ws = list(ev.get("workers") or [])
        return {
            "event": "worker_dispatch",
            "workers": ws,
            "total": ev.get("total", len(ws)),
            "prompt_preview": str(ev.get("prompt", ""))[:TOOL_PREVIEW_LEN],
        }
    if t == "worker_result":
        return {
            "event": "worker_result",
            "worker": ev.get("worker"),
            "index": ev.get("index"),
            "total": ev.get("total"),
            "elapsed_ms": ev.get("elapsed_ms"),
            "ok": bool(ev.get("ok")),
            "preview": str(ev.get("answer", ""))[:TOOL_PREVIEW_LEN],
        }
    if t == "worker_gather_cancelled":
        return {
            "event": "worker_gather_cancelled",
            "completed": list(ev.get("completed") or []),
            "skipped": list(ev.get("skipped") or []),
        }
    # 未识别的直接透传（过滤掉 type 字段，避免 {"event":x, "type":x} 同时残留）
    return {"event": t or "event", **{k: v for k, v in ev.items() if k != "type"}}


def new_session_event(session_id: str) -> dict:
    """Web 建会话的通知事件（结果在协议层响应里，这里只是会话首条流）。"""
    return {"event": "session", "session_id": session_id}


def error_event(message: str) -> dict:
    """单轮运行出错的标准事件。"""
    return {"event": "error", "message": message}
