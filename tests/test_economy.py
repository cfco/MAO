"""省 token 开关（economy，二元）：协议层 added 于 2026-09-23（方案 B）。

覆盖范围（全部离线，无需 API key / 网络）：
- Config 读取：缺省 false、字符串/数字等字面量容错解析、非法值回退默认
- Bridge 初始值：暖启动未配 → false；配置了 → 跟随 cfg.economy
- ready 事件：握手首行必须带出 economy 开关
- set_economy 指令：true/false 切换、非法值保持原值且带 error、响应回显新状态
- health 指令：预检快照顺带回显当前 economy

语义边界（方案 B 红线）：MAO 只负责透传/切换开关，**绝不**自行判断
"分工还是重复验证"——模式判断全部留给外部主。本文件只字不碰模式判断。
"""
from __future__ import annotations

from agent.bridge import _parse_economy
from agent.config import Config, _interpolate


def _cfg(**collab) -> Config:
    """造一个最小 Config：只带必要段，collaboration 按需注入字段。"""
    data = {
        "agents": [],
        "llm": {"base_url": "u", "api_key": "k", "model": "m"},
    }
    if collab:
        data["collaboration"] = collab
    return Config(_interpolate(data))


# ---------------- Config：economy 读取与容错 ----------------

def test_economy_defaults_false_when_absent():
    """未配置 collaboration.economy → false（对外不省 token 的保守默认）。"""
    assert _cfg().economy is False


def test_economy_reads_bool_value_from_collab_cfg():
    """配置 true/false → 原样读出（bool 类型直接透传）。"""
    assert _cfg(economy=True).economy is True
    assert _cfg(economy=False).economy is False


def test_economy_tolerates_string_and_numeric_literals():
    """.env 全是字符串，`COLLAB_ECONOMY=true` 读进来是 "true" 也要能解析。"""
    for raw in ("1", "true", "TRUE", "yes", "on", 1, 1.0):
        assert _cfg(economy=raw).economy is True, f"{raw!r} 应解析为 true"
    for raw in ("0", "false", "FALSE", "no", "off", 0, 0.0):
        assert _cfg(economy=raw).economy is False, f"{raw!r} 应解析为 false"


def test_economy_bad_value_falls_back_to_default():
    """非法值（写成了描述文本等）不崩，回退默认 false。"""
    assert _cfg(economy="省 token 模式").economy is False


# ---------------- Bridge：初始值 ----------------

def test_bridge_economy_defaults_false():
    """未配置时 Bridge 初始 _economy=false。"""
    from agent import bridge as br

    b = br.Bridge(_cfg())
    try:
        assert b._economy is False
    finally:
        b.close()


def test_bridge_economy_follows_config():
    """配置 economy=true → Bridge 初始即打开。"""
    from agent import bridge as br

    b = br.Bridge(_cfg(economy=True))
    try:
        assert b._economy is True
    finally:
        b.close()


# ---------------- ready 事件 ----------------

def test_bridge_ready_event_carries_economy():
    """握手首行必须带出 economy，外部主不用额外问。"""
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "-m", "agent", "bridge"],
        input=json.dumps({"cmd": "ping"}, ensure_ascii=False) + "\n",
        capture_output=True, text=True, encoding="utf-8", cwd=str(root), timeout=120,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines, f"bridge 无输出；stderr={proc.stderr[:400]}"
    first = json.loads(lines[0])
    assert first.get("event") == "ready"
    assert "economy" in first, f"ready 应带 economy 字段，实际 keys：{list(first)}"
    assert first["economy"] in (True, False)


# ---------------- set_economy 指令 ----------------

def test_set_economy_toggles_bool():
    """set_economy 接受 true/false 并返回新状态。"""
    from agent import bridge as br

    b = br.Bridge(_cfg())
    try:
        on = b.handle({"cmd": "set_economy", "value": True})
        assert on["ok"] is True and on["economy"] is True
        off = b.handle({"cmd": "set_economy", "value": False})
        assert off["ok"] is True and off["economy"] is False
    finally:
        b.close()


def test_set_economy_accepts_string_literals():
    """外部主可能以文本形式传 "true"/"1"，同样生效。"""
    from agent import bridge as br

    b = br.Bridge(_cfg())
    try:
        assert b.handle({"cmd": "set_economy", "value": "true"})["economy"] is True
        assert b.handle({"cmd": "set_economy", "value": "0"})["economy"] is False
    finally:
        b.close()


def test_set_economy_bad_value_keeps_current_and_errors():
    """非法值：ok:false + error，且开关保持原值（不静默翻转）。"""
    from agent import bridge as br

    b = br.Bridge(_cfg(economy=True))
    try:
        out = b.handle({"cmd": "set_economy", "value": "省 token"})
        assert out["ok"] is False
        assert "value" in out["error"]
        assert out["economy"] is True, "非法值必须保持原值"
        assert b._economy is True
    finally:
        b.close()


def test_set_economy_affects_health_echo():
    """health 顺带回显 economy，且与 set 后的状态一致。"""
    from agent import bridge as br

    b = br.Bridge(_cfg())
    try:
        b.handle({"cmd": "set_economy", "value": True})
        h = b.handle({"cmd": "health"})
        assert h["ok"] is True
        assert h["economy"] is True
    finally:
        b.close()


# ---------------- _parse_economy 单元 ----------------

def test_parse_economy_unit():
    """解析器边界：None/异常结构返回 None（=非法），正反字面量各归其位。"""
    assert _parse_economy(True) is True
    assert _parse_economy(False) is False
    assert _parse_economy("true") is True
    assert _parse_economy("0") is False
    assert _parse_economy("on") is True
    assert _parse_economy("off") is False
    assert _parse_economy(None) is None
    assert _parse_economy("随便") is None
    assert _parse_economy([1]) is None  # 列表等复杂结构不参与
