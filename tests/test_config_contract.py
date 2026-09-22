"""配置契约测试：确保「用户照 .env.example 填完就能跑」这件事真的成立。

背景（实测暴露的真实缺陷）：
`.env.example` 曾经提供 `LLM_KEY`，而 `config.yaml` 引用的是 `${LLM_API_KEY}` ——
用户老老实实填完 `.env`，`config.llm.api_key` 仍然是空串，兜底单模型永远不可用，
启动还提示"$LLM_API_KEY 未找到"（明明他填了）。变量名对不上是纯契约问题，
单看任一文件都发现不了，只能靠"两边对着比"来防。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent.config import _ENV_PATTERN, Config, _interpolate

ROOT = Path(__file__).resolve().parent.parent
CONFIG_YAML = ROOT / "config.yaml"
ENV_EXAMPLE = ROOT / ".env.example"


def _parse_env_example() -> dict[str, str]:
    """把 .env.example 解析成 {变量名: 值}（忽略注释行与空行）。"""
    out: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, val = s.partition("=")
        out[key.strip()] = val.strip()
    return out


def _config_referenced_vars() -> list[tuple[str, str | None]]:
    """收集 config.yaml **数据部分**引用的 ${VAR}，返回 [(变量名, 默认值或 None)]。

    只扫 yaml.safe_load 解析出来的值（注释天然被排除，避免把注释里的示例当引用）；
    用 finditer 而非 findall：findall 对未参与匹配的可选组返回 ''，
    会把"没有默认值"误判成"默认值为空串"，让下面的契约判断永远为假。
    """
    refs: list[tuple[str, str | None]] = []

    def walk(node) -> None:
        if isinstance(node, str):
            for m in _ENV_PATTERN.finditer(node):
                refs.append((m.group(1), m.group(2)))
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8")))
    return refs


# ---------------- 契约：变量名两边必须对得上 ----------------

def test_env_pattern_distinguishes_missing_default():
    """契约测试的前提：group(2) 为 None 表示"无默认值"，'' 表示"默认值为空串"。

    这条是防"测试本身变成恒真"的护栏——曾经用 findall 写契约测试，
    未参与的可选组返回 ''，"是否存在无默认值的引用"因此永远为假，测试形同虚设。
    """
    ms = list(_ENV_PATTERN.finditer("${A} ${B:-x} ${C:-}"))
    assert [(m.group(1), m.group(2)) for m in ms] == [("A", None), ("B", "x"), ("C", "")]


def test_config_referenced_vars_are_covered_by_env_example():
    """config.yaml 引用的每个 ${VAR}，要么在 .env.example 里有定义，要么带 :-默认值。

    只要有一个"引用了但没定义也没默认"的变量，用户照模板填完就会留下空配置。
    """
    refs = _config_referenced_vars()
    # 护栏：解析出足够多的引用，避免 pattern/解析失效后测试变成空转
    assert len(refs) >= 6, f"只解析到 {len(refs)} 个引用，解析逻辑可能已失效：{refs}"
    assert any(default is None for _, default in refs), (
        "应当存在「无默认值」的引用（如 ${LLM_API_KEY}），否则本测试无法覆盖目标场景"
    )
    defined = set(_parse_env_example())

    uncovered = sorted(k for k, default in refs if default is None and k not in defined)
    assert not uncovered, (
        f"这些变量 config.yaml 引用了，但 .env.example 没提供、也没有 :-默认值：{uncovered}；"
        f"用户照 .env.example 填完它们仍为空。"
    )


def test_env_example_has_no_dead_vars():
    """.env.example 里定义的变量必须真的被 config.yaml 引用（否则是死变量，填了没用）。"""
    referenced = {k for k, _ in _config_referenced_vars()}
    dead = sorted(k for k in _parse_env_example() if k not in referenced)
    assert not dead, f".env.example 里这些变量没有任何地方引用（填了不生效）：{dead}"


def test_filling_env_example_activates_llm_section(monkeypatch):
    """照 .env.example 填完 .env 后，config 的 llm 段与智能体池必须真正生效。

    模板可以把"必填项"留空等你手填（本项目的 .env.example 就把两个 KEY 留空，
    注释要求你粘贴自己的 key），所以这里分两步：
    1) 按模板原样解析 → 有默认值/有值的项必须解析出非空结果；
    2) 再把"必填但留空"的变量补上值 → 必须真落到 config 上。
    第 2 步正是修复前的失败场景：填了变量名对不上的项（LLM_KEY vs LLM_API_KEY），
    值再真也不生效。
    """
    data = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    for key, val in _parse_env_example().items():
        monkeypatch.setenv(key, val)

    cfg = Config(_interpolate(data))
    assert cfg.llm.get("base_url"), "兜底单模型的 base_url 不应为空（模板或默认值应提供）"
    assert cfg.llm.get("model"), "兜底单模型的 model 不应为空"
    assert cfg.agent_profiles, "照 .env.example 填完，智能体池不应为空"

    # 必填变量（config 里无默认值的 ${...}）必须都在模板里有对应行，否则用户无处可填
    declared = set(_parse_env_example())
    required = sorted(k for k, default in _config_referenced_vars() if default is None)
    assert required, "应当存在必填变量（如 ${LLM_API_KEY}），否则本测试失去目标"
    for key in required:
        assert key in declared, f"{key} 在 config.yaml 里必填，但 .env.example 没有对应行"

    # 补上值 → 必须真的生效
    for key in required:
        monkeypatch.setenv(key, f"filled-{key}")
    filled = _interpolate(data)
    assert filled["llm"]["api_key"] == "filled-LLM_API_KEY", (
        f"补上必填变量后 api_key 仍未生效：{filled['llm'].get('api_key')!r}"
    )


# ---------------- ${VAR:-默认值} 插值语义 ----------------

def test_interpolate_default_value_used_when_missing(monkeypatch, capsys):
    """缺变量且带 :-默认值 → 用默认值，且不告警（用户已明确写了兜底）。"""
    monkeypatch.delenv("MAO_ABSENT_VAR", raising=False)
    got = _interpolate("${MAO_ABSENT_VAR:-fallback-url}")
    assert got == "fallback-url"
    assert capsys.readouterr().err == "", "带默认值时不应产生缺失告警"


def test_interpolate_env_wins_over_default(monkeypatch):
    """环境变量存在时优先于默认值，空串也算"存在"（尊重用户显式置空）。"""
    monkeypatch.setenv("MAO_PRESENT_VAR", "real-value")
    assert _interpolate("${MAO_PRESENT_VAR:-fallback}") == "real-value"
    monkeypatch.setenv("MAO_PRESENT_VAR", "")
    assert _interpolate("${MAO_PRESENT_VAR:-fallback}") == ""


def test_interpolate_without_default_still_warns(monkeypatch, capsys):
    """无默认值的缺变量保持原行为：空串 + stderr 告警。"""
    monkeypatch.delenv("MAO_ABSENT_VAR2", raising=False)
    assert _interpolate("a${MAO_ABSENT_VAR2}b") == "ab"
    assert "MAO_ABSENT_VAR2" in capsys.readouterr().err


def test_interpolate_default_with_empty_value(monkeypatch):
    """`${VAR:-}` 是显式的"空默认"，不告警、替换为空串。"""
    monkeypatch.delenv("MAO_ABSENT_VAR3", raising=False)
    assert _interpolate("${MAO_ABSENT_VAR3:-}") == ""


@pytest.mark.parametrize("text,expected", [
    ("${V:-a}b", "ab"),
    ("x${V:-a}y", "xay"),
])
def test_interpolate_default_forms(monkeypatch, text, expected):
    monkeypatch.delenv("V", raising=False)
    assert _interpolate(text) == expected
