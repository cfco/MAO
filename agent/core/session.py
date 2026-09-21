"""会话管理：内存历史 + JSONL 增量落盘（data/sessions/）。

只在代码里维护 JSON Lines 一种格式（一行一条消息）：
- 每 add 一次把消息写入内存历史，并追加到待落盘缓冲（_pending）。
- 落盘做限流：累计达到 FLUSH_BATCH 条、或超过 FLUSH_INTERVAL 秒、
  或显式 flush() 时才真正写盘，避免高频 add（Agent 单轮多工具迭代）
  造成大量零碎磁盘写入。正常退出由 Agent.close() 调 flush() 兜底，
  进程崩溃最多丢失一个限流窗口内的少量消息，可接受。

【设计选择】持久化范围：仅 user 与 assistant 的最终回复。
- tool_calls 段（assistant 带 tool_calls）与 tool 段（role=tool）只在
  Agent.run() 的当轮局部变量 messages 里流动，不写入 Session。
- 后果：会话重启后再调用 Agent.run()，messages_with() 重建的消息列里
  没有 tool 交互痕迹。模型看到的只是「用户问题 → 助手答复」的精简对话。
- 这是有意简化：节省盘空间与 token 预算；多轮上下文靠 assistant 答复里
  已经总结过的内容延续，不依赖原始 tool 调用记录。
- 调试时如需查看完整工具链路，看 stdout / Web SSE 流事件即可，那里有
  tool_start / tool_result 的完整记录。
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
import uuid
from pathlib import Path

from ..config import CONTEXT_SAFETY, ROOT

SESSIONS_DIR = ROOT / "data" / "sessions"
TRASH_DIR = ROOT / "data" / "trash"  # 隔离区：待删文件先移这里，超期确认无问题后再物理删除
EXT = ".jsonl"  # 唯一的落盘格式：JSON Lines
MAX_HISTORY = 60  # 送入模型的最多历史条数（超出丢弃最早的）
TRASH_KEEP_DAYS = 7  # 隔离区文件保留天数，超期自动清空
FLUSH_INTERVAL = 2.0  # 落盘限流：距上次写盘超过该秒数才落盘（低频/跟随者场景）
FLUSH_BATCH = 16  # 落盘限流：待写缓冲累计达到该条数立即落盘（高频场景）

# 会话 id 只允许这几类字符并限制长度：它会被直接拼成落盘文件名，
# 若放开 `/`、`\`、`..`、`:` 就等于把文件名交给外部输入（可穿越出 sessions 目录）。
_SID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def sanitize_session_id(session_id: str | None) -> str | None:
    """校验会话 id 能否安全用作文件名。合法返回规范化后的 id，非法返回 None。

    合法：字母/数字/点/下划线/连字符，长度 1-64。
    显式拒绝 `..`（`a..b` 这种无害，但 `..` 单独出现即为上层目录），
    因此对含 `..` 的 id 一律判非法，避免依赖 Path 的隐式归一。
    调用方（Web / bridge）应把 None 当作用户输入错误直接报错，
    不要静默改写，否则调用方持有的 session_id 与落盘名会不一致。
    """
    sid = (session_id or "").strip()
    if not sid or ".." in sid or not _SID_PATTERN.match(sid):
        return None
    return sid


def _default_session_id() -> str:
    """生成默认会话 id：秒级时间戳 + 4 位随机后缀。

    只用 `%Y%m%d-%H%M%S` 会碰撞：实测同一秒内连开三个 Session，三者拿到完全相同的
    id，于是共用同一个 JSONL 文件——历史互相穿插，`Session.cleanup` 按文件清理时
    还会一次带走多个会话。加随机后缀把碰撞概率压到可忽略（16^4 分之一的同秒概率）。
    """
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


class Session:
    def __init__(
        self,
        session_id: str | None = None,
        flush_batch: int | None = None,
        flush_interval: float | None = None,
    ):
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        # 会话 id 会被直接拼成落盘文件名，这里做兜底校验：正常路径下 Web / bridge
        # 已在边界拒绝非法 id 并返回明确报错；若有非法值仍透传进来，绝不拿它当文件名
        # （否则 session_id="../../x" 会把 JSONL 写到 sessions 目录之外），退回时间戳 id。
        safe_sid = sanitize_session_id(session_id)
        if session_id is not None and safe_sid is None:
            # 走 stderr：bridge 协议与 --stream 的 stdout 是纯 JSON 行，不能被提示混入
            print(f"[警告] 非法 session_id 已拒绝用作文件名，退回时间戳会话：{session_id!r}",
                  file=sys.stderr)
        self.session_id = safe_sid or _default_session_id()
        self._lock = threading.Lock()
        self._pending: list[str] = []  # 已入内存、尚未落盘的 JSON 行
        self._last_flush = time.time()
        self.history: list[dict] = self._load()
        # 落盘限流参数：优先用显式传入（来自 config.session.*），缺省回退到模块常量。
        # 注意：默认回退在 __init__ 体内读取模块名，使测试对常量打 monkeypatch 仍生效。
        self._flush_batch = FLUSH_BATCH if flush_batch is None else flush_batch
        self._flush_interval = FLUSH_INTERVAL if flush_interval is None else flush_interval

    @property
    def _path(self) -> Path:
        # session_id 已在 __init__ 通过 sanitize_session_id 白名单校验（无分隔符/..），
        # 因此这里拼接必然落在 SESSIONS_DIR 内，无需每次落盘再 resolve 一次。
        return SESSIONS_DIR / f"{self.session_id}{EXT}"

    # ---------- 落盘清理 ----------

    @staticmethod
    def cleanup(keep_recent: int = 30) -> int:
        """隔离式清理：只保留最近 keep_recent 个会话，多余的先移入隔离区。

        遵循项目规范18（删除重要文件先隔离，后续确认无问题再手动删除）：
        1) 把最旧的会话移入 data/trash/（不直接删除，出错可找回）。
        2) 顺带清空隔离区中超过 TRASH_KEEP_DAYS 天的文件（视为确认无用）。
        返回本次移入隔离区的文件数。调用时机：CLI / Web / bridge 启动时各清一次。
        """
        TRASH_DIR.mkdir(parents=True, exist_ok=True)

        # 2) 先清空隔离区里超期的旧文件（物理删除只发生在隔离满期后）
        try:
            cutoff = time.time() - TRASH_KEEP_DAYS * 86400
            for stale in TRASH_DIR.glob(f"*{EXT}"):
                if stale.stat().st_mtime < cutoff:
                    stale.unlink(missing_ok=True)
        except OSError:
            pass  # 清空隔离区失败不影响主流程

        # 1) 把最旧的会话移入隔离区
        if not SESSIONS_DIR.exists():
            return 0
        files = sorted(
            SESSIONS_DIR.glob(f"*{EXT}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,  # 最新在前
        )
        moved = 0
        for p in files[keep_recent:]:
            try:
                p.rename(TRASH_DIR / p.name)  # 先隔离，不直接删
                moved += 1
            except OSError:  # 正在被占用等，跳过不拖垮启动
                continue
        return moved

    # ---------- 持久化 ----------

    def _load(self) -> list[dict]:
        """加载 JSONL 历史：逐行解析，坏行跳过不拖垮整个会话。"""
        p = self._path
        if not p.exists():
            return []
        history: list[dict] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict):
                history.append(msg)
        return history

    def add(self, role: str, content: str) -> None:
        """追加一条消息：内存立即更新；磁盘写入走限流缓冲（见 _maybe_flush）。"""
        msg = {"role": role, "content": content}
        self.history.append(msg)
        with self._lock:
            self._pending.append(json.dumps(msg, ensure_ascii=False))
        self._maybe_flush()

    def _maybe_flush(self, force: bool = False) -> None:
        """限流落盘：满足条件才真正写盘。

        - force=True：无条件落盘（正常退出 / 测试用）。
        - 否则：待写条数 >= FLUSH_BATCH，或距上次落盘 >= FLUSH_INTERVAL 才写。
        只追加 _pending 中的新行（O(1) 增量写），不重写全量文件。
        """
        with self._lock:
            if not self._pending:
                return
            now = time.time()
            if (
                not force
                and len(self._pending) < self._flush_batch
                and (now - self._last_flush) < self._flush_interval
            ):
                return
            lines = self._pending
            self._pending = []
            self._last_flush = now
        # 锁外做磁盘 IO，避免阻塞并发 add
        with self._path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def flush(self) -> None:
        """强制把缓冲消息落盘（Agent.close / 进程退出前调用，防数据丢失）。"""
        self._maybe_flush(force=True)

    # ---------- 供主循环拼装 ----------

    @staticmethod
    def _est_tokens(text: str) -> int:
        """粗略按字符估算 token 数：CJK 1 字≈1 token，ASCII ≈4 字符≈1 token。"""
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        other = len(text) - cjk
        return cjk + other // 4 + 1

    def messages_with(self, system_prompt: str, context_length: int = 128_000) -> list[dict]:
        """拼装完整消息列表：system + 按上下文窗口裁剪后的历史。

        历史按 token 估算裁剪（而非死条数）：从最早的消息起丢弃，
        直到剩余历史估算 token ≤ context_length×0.75，给回复/工具调用留余量。
        至少保留最近 MAX_HISTORY 条，避免长工具结果把全场清空。
        """
        msgs: list[dict] = [{"role": "system", "content": system_prompt}]
        budget = max(MAX_HISTORY, int(context_length * CONTEXT_SAFETY))
        # 从尾部往回累计，头部溢出即停
        # MAX_HISTORY * 4：扩大候选窗口，避免长工具结果导致候选被过早截断
        kept: list[dict] = []
        used = 0
        for msg in reversed(self.history[-MAX_HISTORY * 4 :]):
            est = self._est_tokens(str(msg.get("content", "")))
            if used + est > budget and used > 0:
                break
            kept.append(msg)
            used += est
        kept.reverse()
        msgs.extend(kept)
        return msgs
