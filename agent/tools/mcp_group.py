"""MCP 连接组的进程内共享（按 servers 配置签名复用 + 引用计数）。

为什么独立成模块（2026-11-04，自 mcp_client.py 拆出守「单文件 ≤500 行」约定）：
mcp_client.py 专注"一条连接怎么建立/重连/调用"（McpConnection），本模块专注
"一组连接怎么被多个调用方安全复用"。组段历史注释一并迁入。
"""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING

from .base import ToolRegistry
from .mcp_client import McpConnection, _safe_name, connect_and_register, stop_all  # noqa: F401

if TYPE_CHECKING:  # 仅类型标注：运行期不引入（避免与 mcp_client 成环）
    from .mcp_client import McpTool


# ---------- 进程内共享连接组（按 servers 配置签名复用） ----------


class McpGroup:
    """一组进程内共享的 MCP 长连接 + 已发现的工具对象，带引用计数。

    为什么要共享：同一进程（CLI / bridge / 外部主）多次派工或并发调用工具时，
    各自 connect_and_register 会为同一批 server 反复建立连接（stdio 场景 =
    反复拉起同样的子进程），且每个 server 首次连接最长要等 90s。共享后同一进程
    只维护一套连接，新调用方只做"把自己的工具注册表挂上去"这一件轻活。McpTool
    本身无状态（只持有 conn 引用），可安全地同时注册到多个调用方的注册表里。

    引用计数：最后一个使用者 release 时才真正 stop_all —— 否则一个调用方关闭就会
    把别的调用方正在用的连接一起掐掉。
    """

    def __init__(self, server_cfgs: list[dict], call_timeout: float = 300.0):
        self.server_cfgs = server_cfgs
        self.call_timeout = max(1.0, float(call_timeout))
        self.conns: list[McpConnection] = []
        self.tools: list[McpTool] = []
        self.status: list[str] = []
        self._lock = threading.Lock()
        self._refs = 0
        self._closed = False

    def start(self) -> None:
        """建立连接并发现工具（只在首次创建该组时调用一次）。"""
        collector = ToolRegistry()
        self.status = connect_and_register(
            self.server_cfgs,
            collector,
            self.conns,
            out_tools=self.tools,
            call_timeout=self.call_timeout,
        )

    def try_acquire(self) -> bool:
        """登记一个使用者；组已关闭则返回 False，由调用方重建（不抛）。

        为什么不再抛 RuntimeError：唯一调用点在 `Bridge.__init__`，而 `Bridge(cfg)` 在
        `serve()` 的 try 之外 —— 一个"刚被别人用完并关闭"的组让 bridge 在发出 ready 行之前就
        崩掉，外部驱动方看到的是空 stdout。关组与登记本来就是两条锁上的竞争，该由取组方兜住。
        """
        with self._lock:
            if self._closed:
                return False
            self._refs += 1
            return True

    def release(self) -> None:
        """释放一个使用者；引用归零时关闭全部连接。"""
        with self._lock:
            self._refs -= 1
            if self._refs > 0:
                return
            self._closed = True
        stop_all(self.conns)

    @property
    def refs(self) -> int:
        with self._lock:
            return self._refs


_groups_lock = threading.Lock()
_groups: dict[str, McpGroup] = {}


def _group_key(server_cfgs: list[dict], call_timeout: float = 300.0) -> str:
    """配置签名：内容一致（含键序无关）即同一组，可复用同一批连接。

    超时也进签名：同配置不同超时不该共用连接，否则先到者的超时悄悄决定后到者。
    """
    return (
        json.dumps(server_cfgs, sort_keys=True, ensure_ascii=False, default=str)
        + f"|call_timeout={call_timeout}"
    )


def acquire_group(server_cfgs: list[dict], call_timeout: float = 300.0) -> McpGroup:
    """取得（或建立）与配置对应的共享连接组，并登记一个使用者。

    建连在全局锁外完成：单个 server 首连最长 90s，占着锁会把别的会话一起堵住。
    """
    key = _group_key(server_cfgs, call_timeout)
    while True:
        with _groups_lock:
            group = _groups.get(key)
            if group is not None:
                # 复用判定与引用登记必须在同一把表锁内完成：旧写法先取组、再到锁外 acquire()，
                # 这中间另一个会话把引用降到 0 并关组，就会出现"拿到一个已关闭的组"。
                if group.try_acquire():
                    return group
                _groups.pop(key, None)  # 已关闭的组不能复用，摘掉后重建
        fresh = McpGroup(list(server_cfgs), call_timeout)
        fresh.start()  # 锁外建连（首连可能耗时 90s，绝不占着全局锁）
        with _groups_lock:
            if key not in _groups:  # 并发下别人已建好就用先到的那份
                _groups[key] = fresh
                fresh.try_acquire()
                return fresh
        stop_all(fresh.conns)


def release_group(group: McpGroup) -> None:
    """释放一个使用者；引用归零时关闭连接并从共享表移除。"""
    group.release()
    if group.refs > 0:
        return
    with _groups_lock:
        for key, g in list(_groups.items()):
            if g is group:
                _groups.pop(key, None)


def register_group_tools(registry: ToolRegistry, group: McpGroup) -> None:
    """把共享组已发现的工具注册进指定注册表（同批 McpTool 对象跨会话复用）。"""
    for tool in group.tools:
        if registry.get(tool.name) is None:
            registry.register(tool)
