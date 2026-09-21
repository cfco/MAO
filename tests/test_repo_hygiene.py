"""仓库卫生守卫：`.gitignore` 规则必须真的生效、lint 不得扫到隔离区。

两条背景都是本项目实际发生过的：
1. `.gitignore` 不支持行内注释 —— `data/    # 会话历史` 会被当成一个字面 pattern，
   整条规则失效，于是 `data/` 从未被忽略、会话历史（含对话内容）会被提交。
2. `[tool.ruff] exclude` 是**替换**语义，会顶掉 ruff 的默认排除表
   （.venv/.git/__pycache__/.pytest_cache… 全部重新进入扫描范围），
   只能靠 `.gitignore` 兜着。两处一起坏时，lint 就会去扫隔离区的旧文件而让 CI 变红。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GITIGNORE = ROOT / ".gitignore"
PYPROJECT = ROOT / "pyproject.toml"


def _gitignore_rules() -> list[str]:
    """返回 .gitignore 里的**规则行**（去掉注释行与空行，含以 ! 开头的取反规则）。"""
    rules = []
    for raw in GITIGNORE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            rules.append(line)
    return rules


def test_gitignore_rules_are_standalone_lines():
    """.gitignore 规则行里不得出现 #：git 不支持行内注释，写了会让整条规则失效。"""
    bad = [r for r in _gitignore_rules() if "#" in r]
    assert not bad, (
        f"这些 .gitignore 规则被行内注释破坏（git 会当字面 pattern，规则等于没写）：{bad}"
        "——把说明移到规则上一行。"
    )


@pytest.mark.parametrize("needed", ["data/", ".venv", ".env"])
def test_sensitive_paths_have_rules(needed):
    """会话历史 / 虚拟环境 / 密钥文件必须有明确的忽略规则。"""
    assert needed in _gitignore_rules(), (
        f"{needed} 缺少独立忽略规则；当前规则：{_gitignore_rules()}"
    )


@pytest.mark.parametrize("path", ["data/sessions/xxx.jsonl", ".env", ".venv/lib/x.py"])
def test_gitignore_actually_ignores_sensitive_paths(path):
    """用 git 自己判定规则真的生效（只看规则文本会漏掉"写了但不匹配"的情况）。"""
    if shutil.which("git") is None:
        pytest.skip("环境无 git，跳过")
    r = subprocess.run(["git", "check-ignore", "-q", path], cwd=str(ROOT),
                       capture_output=True, text=True)
    if r.returncode not in (0, 1):
        pytest.skip(f"git 不可用（rc={r.returncode}）")
    assert r.returncode == 0, f"{path} 未被 .gitignore 忽略"


def test_ruff_extends_default_excludes_instead_of_replacing():
    """ruff 配置必须用 extend-exclude，且显式排除隔离区 data/。

    用 `exclude` 会把 ruff 默认排除表整体替换掉，.venv 等目录重新被扫描，
    于是 lint 的通过与否取决于 .gitignore 是否写对——这种隐性耦合不能再留。
    """
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+；本项目开发解释器为 3.14
    ruff_cfg = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["ruff"]

    assert "exclude" not in ruff_cfg, (
        "请用 extend-exclude：exclude 会顶掉 ruff 默认排除表，让 .venv/.git 等重新被扫描"
    )
    assert "data" in ruff_cfg.get("extend-exclude", []), (
        "隔离区/运行时产物 data/ 必须显式排除，否则 data/trash 里的文件会被 lint"
    )
