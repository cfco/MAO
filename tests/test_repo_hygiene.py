"""仓库卫生守卫：`.gitignore` 规则必须真的生效、lint 不得扫到隔离区、模板不得带值。

前三条背景都是本项目实际发生过的，第四条是防漂移的守卫：
1. `.gitignore` 不支持行内注释 —— `data/    # 会话历史` 会被当成一个字面 pattern，
   整条规则失效，于是 `data/` 从未被忽略、会话历史（含对话内容）会被提交。
2. `[tool.ruff] exclude` 是**替换**语义，会顶掉 ruff 的默认排除表
   （.venv/.git/__pycache__/.pytest_cache… 全部重新进入扫描范围），
   只能靠 `.gitignore` 兜着。两处一起坏时，lint 就会去扫隔离区的旧文件而让 CI 变红。
3. `.env` 是唯一承载端点/密钥的文件，此前没有 `.env.example`：新环境只能靠读代码
   猜变量名。模板一旦带了真实值就等于把密钥提交进公开仓库，故这里守"模板必须全注释、
   且列出的变量名都还有效"。
4. 版本号手写两份（`pyproject.toml` 的 `version` / `agent/__init__.py` 的
   `__version__`），改一处忘一处不会报错，只会让外部主 `ping` 到的版本与装包元数据不符。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from agent.config import _load_dotenv

ROOT = Path(__file__).resolve().parent.parent
GITIGNORE = ROOT / ".gitignore"
PYPROJECT = ROOT / "pyproject.toml"
ENV_EXAMPLE = ROOT / ".env.example"
CONFIG_YAML = ROOT / "config.yaml"


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


def _check_ignore_isolated(tmp_path: Path, path: str) -> bool:
    """在**只装本项目 .gitignore 的空仓**里问 git：这个路径算不算被忽略。

    为什么不直接在项目根跑 `git check-ignore`（本测试曾经就是这么写的，会假红）：
    git 的 exclude 规则来自仓库内部，但判定结果却受"这个仓库放在哪儿"牵连——把仓库
    解压/克隆到某个会被外层规则命中的目录时（最典型的就是本仓库根 .gitignore 自己带
    `.tmp_*`，把工作区导出到 `.tmp_verify/` 再跑），`.env.example` 会被**外层目录规则**
    判成"被忽略"而假红；反方向同样危险：`data/` 之类若被外层规则命中，即使我们这份
    规则写坏了也照样判"已忽略"，那是假绿。

    空仓里 repo 之外的 .gitignore / info/exclude 都不参与判定，结果只取决于我们这一份
    规则文本；系统级与用户级 exclude 也一并隔离（GIT_CONFIG_NOSYSTEM + 把 HOME 指到
    临时目录）。
    """
    if shutil.which("git") is None:
        pytest.skip("环境无 git，跳过")
    repo = tmp_path / "ignore-probe"
    repo.mkdir(exist_ok=True)  # 一个用例里可能正反各问一次，探针仓复用
    (repo / ".gitignore").write_text(GITIGNORE.read_text(encoding="utf-8"), encoding="utf-8")
    env = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": str(repo),
        "USERPROFILE": str(repo),
        "XDG_CONFIG_HOME": str(repo),
    }
    init = subprocess.run(
        ["git", "init", "-q"], cwd=str(repo), env=env, capture_output=True, text=True
    )
    if init.returncode != 0:
        pytest.skip(f"git init 不可用（rc={init.returncode}）：{init.stderr.strip()}")
    r = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode not in (0, 1):
        pytest.skip(f"git check-ignore 不可用（rc={r.returncode}）：{r.stderr.strip()}")
    return r.returncode == 0


@pytest.mark.parametrize(
    "path", ["data/sessions/xxx.jsonl", ".env", ".venv/lib/x.py", "data/model_health.json"]
)
def test_gitignore_actually_ignores_sensitive_paths(tmp_path, path):
    """用 git 自己判定规则真的生效（只看规则文本会漏掉"写了但不匹配"的情况）。"""
    assert _check_ignore_isolated(tmp_path, path), f"{path} 未被本项目的 .gitignore 忽略"


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


# ---------------- .env.example：新环境照着填的那张单子 ----------------


def _env_example_lines() -> list[str]:
    assert ENV_EXAMPLE.exists(), ".env.example 缺失：新环境只能靠读代码猜变量名"
    return ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()


def test_env_example_is_committable_while_env_is_not(tmp_path):
    """模板要能提交（它是文档），真实 .env 必须被忽略 —— 两条方向都在空仓里判定。

    正反都断言：只查 `.env.example` 不被忽略，`.env` 那行规则被人写坏（比如误删）时
    这条守卫仍然绿；只查 `.env` 被忽略，则有人加了 `.env.*` 一刀切把模板也吃掉时不会报。
    判定走 `_check_ignore_isolated`，因此把仓库解压到 `.tmp_*` 这类会被外层规则命中的
    目录里也不会假红。
    """
    assert _check_ignore_isolated(tmp_path, ".env"), (
        ".env 没被忽略：真实端点/密钥会被提交进公开仓库"
    )
    assert not _check_ignore_isolated(tmp_path, ".env.example"), (
        ".env.example 被 .gitignore 吃掉了（`.env.*` 之类的规则没被 ! 取反）——"
        "它是新环境照着填的模板，必须能提交"
    )


def test_env_example_injects_nothing(tmp_path, monkeypatch):
    """模板必须**整份是注释**：复制成 .env 后不填值也不该往环境里注任何东西。

    留成 `KEY=`（空值）会被 _load_dotenv 当真值注入：config.yaml 的 `${KEY:-默认}`
    会被空串顶掉，数值项退化成"警告 + 默认"，端点项变成空 base_url —— 用户看到的是
    "照着模板复制却不能跑"。
    """
    for k in ("NODE_A_ENDPOINT", "COLLAB_MAX_WORKERS"):
        monkeypatch.delenv(k, raising=False)
    active = [ln for ln in _env_example_lines() if ln.strip() and not ln.strip().startswith("#")]
    assert not active, f".env.example 里有未注释的行（复制即注入）：{active}"
    assert _load_dotenv(ENV_EXAMPLE, set()) == [], "模板解析后不应注入任何变量"


def test_env_example_names_are_still_real_knobs():
    """模板列的变量名必须还在用：config.yaml 的 ${VAR} 插值或代码直接读的环境变量。

    名字改名/废弃后模板不会自己报警，新人照着填一个失效的变量，配置静默不生效。
    """
    yaml_text = CONFIG_YAML.read_text(encoding="utf-8")
    code_text = "\n".join(p.read_text(encoding="utf-8") for p in [*(ROOT / "agent").rglob("*.py")])
    listed = [
        m.group(1) for ln in _env_example_lines() if (m := re.match(r"\s*#([A-Z][A-Z0-9_]*)=", ln))
    ]
    assert listed, "模板里没解析到任何变量名（格式应为 `#KEY=值或说明`）"
    stale = [
        n
        for n in listed
        if f"${{{n}}}" not in yaml_text
        and f"${{{n}:-" not in yaml_text
        and f'"{n}"' not in code_text
    ]
    assert not stale, f"这些变量在 config.yaml 与代码里都已不被引用，模板应删掉：{stale}"
    # 反向也要成立：config.yaml 里"只能来自 .env"的端点/密钥类变量都得在模板出现
    must = [
        v
        for v in re.findall(r"\$\{([A-Z][A-Z0-9_]*)\}", yaml_text)
        if v.endswith(("_ENDPOINT", "_KEY", "_BASE_URL", "_TOKEN"))
    ]
    missing = [v for v in must if v not in listed]
    assert not missing, f"config.yaml 引用了这些端点/密钥变量但模板没列：{missing}"


def test_version_is_declared_consistently():
    """两份手写版本号必须一致：`ping`/`ready` 回给外部主的是 __version__，装包元数据是 pyproject。

    漂移不会报错，只会让外部主看到的版本是假的。这里不用 tomllib（3.11+ 才有，
    requires-python 的下限是 3.10，CI 的 3.10 腿会直接 ImportError），做行级正则就够。
    """
    pyproject_text = PYPROJECT.read_text(encoding="utf-8")
    init_text = (ROOT / "agent" / "__init__.py").read_text(encoding="utf-8")
    pkg = re.search(r'^version = "([^"]+)"', pyproject_text, re.M)
    code = re.search(r'^__version__ = "([^"]+)"', init_text, re.M)
    assert pkg and code, "找不到版本号声明处（写法变了就同步改这个守卫）"
    assert pkg.group(1) == code.group(1), (
        f"版本漂移：pyproject.toml={pkg.group(1)}，agent/__init__.py={code.group(1)}"
    )
