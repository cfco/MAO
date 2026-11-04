"""发现适配器：每站一个，知道怎么拉在线免费清单、怎么解析。

为什么独立成模块（2026-11-04，自 node_discovery.py 拆出守「单文件 ≤500 行」约定）：
适配器（怎么连远端、怎么解析响应、怎么清洗不可信 id）与对账/调度（node_discovery.py）
是两种职责；新增一个站 = 在这里加一个 Adapter 子类，不动调度与叠加层逻辑。
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class FreeModel:
    """一个在线免费模型（bare 名 + 上下文窗口，窗口 0=未知）。"""

    model: str
    ctx: int = 0


# ---------- HTTP 抓取（stdlib urllib，不引入新依赖） ----------


def _http_get_json(url: str, api_key: str, timeout: float) -> Any | None:
    """GET 一个 JSON 接口；任何异常（网络/鉴权/解析）返回 None（调用方据此跳过）。"""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - 仅 HTTPS 受信端点
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


# 远端模型 id 的可接受字符：只认"看着像模型名"的串。
# 逗号是清单的分隔符（`_split_items` 按它切），空白会顶坏 `name@ctx` 的拆分 ——
# 一个含逗号的远端 id 写进叠加层就等于往 env 里注入多条模型（下游按条建 worker）。
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,119}$")


def _as_int(v: Any) -> int:
    """尽力把各种形式的窗口/context_length 转成 int；失败返回 0（未知）。"""
    if v is None:
        return 0
    if isinstance(v, bool):
        return 0
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    s = str(v).strip().lower().rstrip("k")
    try:
        return int(float(s)) if "." in str(v) else int(s)
    except ValueError:
        return 0


class Adapter:
    """发现适配器协议：fetch 返回在线免费模型列表；返回 None=失败/跳过。

    `host` 是本站必须落在的域：适配器写死自己那个公开网关时，若不给站加归属校验，
    任何标了该适配器名的站都会把**别人的清单**叠进自己的 env（跑一轮就派工到本站
    根本没有的模型）。`free_marker` 限定"哪些条目归本适配器管"：叠加层的 retired
    只能作用于免费档条目，否则人工加进 registry 的付费模型会因为"不在线"被下架。
    """

    name = "base"
    host = ""
    free_marker = ""

    def fetch(self, base_url: str, api_key: str, timeout: float) -> list[FreeModel] | None:
        raise NotImplementedError


# 两个校验都做成模块函数并按属性读取：调用方（含测试里的假适配器）常常不继承 Adapter，
# 走 `adapter.owns_station(...)` 会对这类对象抛 AttributeError，把整轮发现打成 fetch_failed。
def _owns_station(adapter: Any, base_url: str) -> bool:
    """本站端点是否属于该适配器（`host` 子串比对；未声明 host 的适配器不受限）。"""
    host = getattr(adapter, "host", "") or ""
    return not host or host.lower() in (base_url or "").lower()


def _manages(adapter: Any, model_id: str) -> bool:
    """该 registry 条目是否归本适配器对账（免费档标记为空 = 全部归它管）。"""
    marker = getattr(adapter, "free_marker", "") or ""
    return not marker or model_id.endswith(marker)


def _clean_models(live: list[FreeModel]) -> list[FreeModel]:
    """丢掉不可信远端清单里"不像一个模型名"的条目（见 _MODEL_ID_RE 的注入面）。"""
    return [m for m in live if _MODEL_ID_RE.fullmatch(m.model or "")]


class KiloAdapter(Adapter):
    """kilo.ai 公开网关：无需登录即可列全部模型，`:free` 后缀即免费档。"""

    name = "kilo"
    host = "api.kilo.ai"
    free_marker = ":free"

    def fetch(self, base_url: str, api_key: str, timeout: float) -> list[FreeModel] | None:
        data = _http_get_json("https://api.kilo.ai/api/gateway/models", "", timeout)
        if not isinstance(data, dict):
            return None
        items = data.get("data") or data.get("models") or []
        out: list[FreeModel] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            mid = it.get("id") or it.get("name")
            if not mid or not str(mid).endswith(self.free_marker):
                continue
            out.append(FreeModel(str(mid), _as_int(it.get("context_length"))))
        return out or None
