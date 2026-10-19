"""LLM 适配层：统一走 OpenAI 兼容协议（chat.completions）。

DeepSeek / Kimi / 豆包(方舟) / OpenAI / 本地 Ollama 均兼容，换模型只改配置。

免费节点友好设计（"单节点故障不能影响整体运行"）：
- 每次请求带超时（默认 120s），不挂死等网络黑洞。
- 分类捕获异常：限流(429)/连接失败/超时/认证/其它。
- 可重试类错误（限流、5xx、超时、连接）自动指数退避重试
  （限流退避更长，给免费节点的速率限制让路），证书类错误不重试。
- 抛出的 LLMError 带 error_type / retryable，上层（WorkerPool）据此做
  节点冷却等编排：单节点抖了，换/跳过其它节点，整体不受影响。
"""

from __future__ import annotations

import random
import threading
import time

import openai
from openai import OpenAI

try:  # 兼容不同 openai 包版本
    from openai import (
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        RateLimitError,
    )
except ImportError:  # pragma: no cover - 老版本 openai 防御
    APIConnectionError = openai.APIConnectionError
    APITimeoutError = openai.APITimeoutError
    AuthenticationError = openai.AuthenticationError
    RateLimitError = openai.RateLimitError

from .redact import safe_error_text

DEFAULT_TIMEOUT = 120
DEFAULT_RETRIES = 2
# 限流/连接类错误的初始退避秒数（指数增长，封顶 60s；限流更慢）
_BASE_BACKOFF_RATE_LIMIT = 8.0
_BASE_BACKOFF_OTHER = 2.0
_MAX_BACKOFF = 60.0
# 除 5xx 外这些 HTTP 状态码也值得退避重试：408 请求超时 / 425 Too Early /
# 429 限流（SDK 一般已归到 RateLimitError，这里是兜底）。免费中转网关超时很常见，
# 一次失败就放弃会让"单节点抖动不影响整体"的承诺打折。
_RETRYABLE_STATUS = frozenset({408, 425, 429})

# 错误类型常量
ERR_TIMEOUT = "timeout"
ERR_RATE_LIMIT = "rate_limit"
ERR_AUTH = "auth"
ERR_UNAVAILABLE = "unavailable"
ERR_API = "api"
ERR_OTHER = "other"


def worst_case_per_request(timeout: float, max_retries: int) -> float:
    """一次 `chat()` 调用的最坏墙钟耗时上界（秒）：所有尝试 + 尝试之间的退避。

    编排层的整体限时必须按这个数折算，而不是按 `timeout` 单次：429/超时/5xx 会在
    `chat()` 内部退避重试，一次派工实际可能占用 `timeout x (retries+1)` 再加退避秒数。
    按单次 timeout 折算会把批次时间算短，尾部批次被判"未完成"丢弃 —— 额度已经烧掉、
    结果却拿不到（与 2026-09-22 那条「池大 + 并发低 + 固定限时」的 P0 同一族）。
    """
    attempts = 1 + max(0, int(max_retries))
    return attempts * timeout + (attempts - 1) * (_MAX_BACKOFF + 1.0)  # +1.0 = jitter 上界


def backoff_seconds(attempt: int, error_type: str) -> float:
    """指数退避秒数：2^attempt x 基数 + jitter，**总时长封顶 `_MAX_BACKOFF`**。

    jitter 加在 `min()` 之内（旧写法 `min(cap, x) + uniform(0,1)` 会让"封顶"多出 1 秒，
    把上界当硬约束的调用方（如整体限时折算）算错）。限流类基数更大，给免费节点的
    速率限制让路。
    """
    base = _BASE_BACKOFF_RATE_LIMIT if error_type == ERR_RATE_LIMIT else _BASE_BACKOFF_OTHER
    return min(_MAX_BACKOFF, (2**attempt) * base + random.uniform(0, 1.0))


class LLMError(Exception):
    """LLM 调用失败（已分类）。

    retryable=True 表示值得退避重试（限流/超时/5xx/连接）；
    retryable=False 表示重试也没用（认证失败等），上层直接判失败。
    """

    def __init__(self, error_type: str, message: str, retryable: bool = True):
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable


def _first_choice(resp):
    """取响应里的第一条 choice，取不到就抛可重试的 LLMError。

    免费中转站常见两种畸形响应：内容被过滤后返回 `choices: []`，或字段缺失。
    早期实现直接 `resp.choices[0]`，这会以裸 IndexError 逃出 LLMClient，
    绕过上层的错误分类与退避重试（主智能体只捕获 LLMError），一路冒到调用方。
    在这里显式归类为 service 端错误，让它走正常的重试/冷却编排。
    """
    choices = getattr(resp, "choices", None) or []
    if not choices:
        raise LLMError(ERR_API, "服务端返回空结果（choices 为空），可能是内容过滤或节点异常")
    return choices[0].message


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_RETRIES,
    ):
        # 关闭 SDK 内置重试：让 chat() 的自定义指数退避成为唯一重试源，
        # 避免两层退避叠加导致实际重试次数/超时不可预期。
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", max_retries=0)
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        # 在飞计数 + 延迟关闭（供 WorkerPool 回收被移出工人的连接池）：
        # 直接 close 会掐断正在飞的请求；完全不关则每次热同步都漏一个 httpx 池。
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        self._close_pending = False
        self._closed = False

    # ---------- 对外接口 ----------

    def chat_once(self, messages: list[dict], max_tokens: int | None = None) -> dict:
        """**一次**网络尝试：不重试、不睡眠。

        派工路径要的是这个粒度：同站串行门只该包住真实的一次往返，退避睡眠绝不能
        占着门（见 station_gate 与 `WorkerPool._call_result`）——否则一个坏模型会把整站
        锁住 `timeout x (retries+1) + 退避` 量级，同站其它可用模型全排在后面。
        进出都记在飞数：`close_when_idle` 靠它判断"现在关会不会掐断请求"。
        `max_tokens` 供延迟探测限制输出长度（让样本只反映往返速度、不混入"模型话多"）。
        """
        self._enter_call()
        try:
            return self._once(messages, max_tokens=max_tokens)
        finally:
            self._leave_call()

    def chat(self, messages: list[dict]) -> dict:
        """发起一轮对话补全（内部按 `max_retries` 退避重试），返回 assistant 消息 dict。

        返回 {"role": "assistant", "content": "..."}。工人池是纯文本执行器，本适配层
        不启用 function calling（无工具入参、不解析 tool_calls）。
        失败：抛 LLMError（可重试错误已在此自动退避重试）。
        用途：不经站点门的调用方（延迟探测客户端本来就是 `max_retries=0`，一次就回）。
        真实派工不走这里，走 `chat_once` + 编排层的门外重试环。
        """
        last: LLMError | None = None
        self._enter_call()  # 整条重试链算一次"在飞"：中途被回收会打断退避后的重试
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    return self._once(messages)
                except LLMError as e:
                    last = e
                    if not e.retryable or attempt >= self.max_retries:
                        raise
                    time.sleep(self._backoff(attempt, e.error_type))  # 经实例钩子：可被测试替换
            raise last  # 理论不可达（上面必 raise），占位保险
        finally:
            self._leave_call()

    def close(self) -> None:
        """释放底层 HTTP 连接池（幂等，可重复调用）。

        OpenAI 客户端持有一个 httpx 连接池：不显式关，就只能等 GC 回收，
        长生命周期进程（bridge 常驻、反复 refresh_nodes）会积压废弃连接。
        释放失败不应影响调用方收尾，故整体吞异常；`_closed` 让重复调用变成空操作。
        """
        with self._inflight_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.client.close()
        except Exception:  # noqa: BLE001 - 释放失败不影响上层收尾
            pass

    def close_when_idle(self) -> None:
        """标记"不再复用"，等在飞请求收尾后再关连接池（空闲则当场关）。

        用于工人被移出池（`sync_free_nodes`）：此刻它可能正在回答一个已发出的请求，
        直接关会把在飞请求打成异常；完全不关则连接池挂到进程退出（频繁热同步会累积）。
        标记后仍接受新调用（旧引用可能还在回调链里），最后一次收尾时统一关闭——
        `close()` 幂等，重复触发无害。
        """
        with self._inflight_lock:
            self._close_pending = True
            idle = self._inflight <= 0
        if idle:
            self.close()

    def _enter_call(self) -> None:
        with self._inflight_lock:
            self._inflight += 1

    def _leave_call(self) -> None:
        with self._inflight_lock:
            self._inflight -= 1
            should_close = self._inflight <= 0 and self._close_pending
        if should_close:
            self.close()

    # ---------- 内部 ----------

    def _once(self, messages: list[dict], max_tokens: int | None = None) -> dict:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
        if max_tokens is not None:
            # 只在显式要求时带上：免费网关对未知参数行为不一（有的直接 400），
            # 派工路径保持"只发标准三件套"，探测路径由 LatencyMixin 按站决定是否带。
            kwargs["max_tokens"] = max_tokens
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except APITimeoutError as e:
            raise LLMError(ERR_TIMEOUT, f"请求超时（{self.timeout}s）", retryable=True) from e
        except RateLimitError as e:
            # 免费节点最常见的限流：视为可重试，退避更久
            raise LLMError(ERR_RATE_LIMIT, "触发速率/用量限制(429)，请稍后再试") from e
        except AuthenticationError as e:
            raise LLMError(ERR_AUTH, "API key 无效或被拒绝", retryable=False) from e
        except APIConnectionError as e:
            raise LLMError(ERR_UNAVAILABLE, "连接失败（节点不可达）") from e
        except openai.APIStatusError as e:
            code = e.status_code
            if code >= 500 or code in _RETRYABLE_STATUS:
                # 5xx 是服务端问题；408/425/429 属于"再试一次可能就好"的临时状态
                raise LLMError(ERR_API, f"服务端错误 HTTP {code}") from e
            # 其余 4xx 是客户端问题（参数错/鉴权/资源不存在等），重试没用
            raise LLMError(
                ERR_API, f"API 错误 HTTP {code}: {safe_error_text(e.message)}", retryable=False
            ) from e
        except Exception as e:  # noqa: BLE001 - 兜底：未知错误保守判定为不可重试
            # 兜底分支只能透传异常原文，而底层异常常回显请求 URL：key 放 query 里的
            # 端点会从这里泄进 stdout 协议行与工人上下文，故先过一遍脱敏（见 redact.py）
            raise LLMError(
                ERR_OTHER, f"{type(e).__name__}: {safe_error_text(e)}", retryable=False
            ) from e
        msg = _first_choice(resp)
        return {"role": "assistant", "content": msg.content or ""}

    @staticmethod
    def _backoff(attempt: int, error_type: str) -> float:
        """兼容旧调用点的别名：真正的退避函数是模块级 `backoff_seconds`。"""
        return backoff_seconds(attempt, error_type)
