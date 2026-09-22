"""工作区（tools.workspace）能力与安全边界的回归测试。

背景：工具层原先把所有路径硬绑在 MAO 项目根上，MAO 因此无法用于驱动外部项目
——主智能体连目标项目的一个文件都读不到，"驱动外部项目"成了空话。
本测试守两件事：
1) 配了 workspace 后，外部目录可读、可列、可当 cwd；
2) 没配（或配了之外）的路径，越界依然被拒 —— 安全边界不能因为这次放宽而消失。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools import builtin


@pytest.fixture(autouse=True)
def _reset_roots():
    """每个用例前后都恢复默认根。全局状态必须清干净，否则会污染其它测试。"""
    builtin.set_workspace_roots([])
    yield
    builtin.set_workspace_roots([])


class _FakeCfg:
    """build_builtin_tools 只用到这两个属性。"""

    def __init__(self, workspace):
        self.workspace = workspace
        self.shell_timeout = 10


def test_default_limits_to_project_root():
    """不配 workspace 时，只有项目根可访问。"""
    roots = builtin.set_workspace_roots([])
    assert roots == (builtin.ROOT_RESOLVED,)
    assert builtin._resolve("config.yaml") is not None          # noqa: SLF001
    outside = builtin.ROOT_RESOLVED.parent / "definitely-outside.txt"
    assert builtin._resolve(str(outside)) is None               # noqa: SLF001


def test_workspace_extends_reachable_roots(tmp_path):
    """配了 workspace，外部目录内路径可解析。"""
    extra = tmp_path / "extproj"
    extra.mkdir()
    target = extra / "a.txt"
    target.write_text("hi", encoding="utf-8")

    roots = builtin.set_workspace_roots([str(extra)])
    assert len(roots) == 2
    assert builtin._resolve(str(target)) == target.resolve()    # noqa: SLF001


def test_path_outside_configured_workspace_still_blocked(tmp_path):
    """放宽后仍必须挡住「配了 A、却去访问 B」的情况。"""
    allowed = tmp_path / "allowed"
    other = tmp_path / "other"
    allowed.mkdir()
    other.mkdir()
    (other / "x.txt").write_text("secret", encoding="utf-8")

    builtin.set_workspace_roots([str(allowed)])
    assert builtin._resolve(str(other / "x.txt")) is None       # noqa: SLF001


def test_nonexistent_workspace_entry_is_skipped(tmp_path):
    """配错路径不该让工具全挂 —— 该条直接跳过。"""
    roots = builtin.set_workspace_roots([str(tmp_path / "not-exist")])
    assert roots == (builtin.ROOT_RESOLVED,)


def test_relative_workspace_resolves_against_root():
    """相对路径基于项目根解析，"." 不会引入重复根。"""
    assert builtin.set_workspace_roots(["."]) == (builtin.ROOT_RESOLVED,)


def test_backup_of_external_file_is_prefixed(tmp_path):
    """外部项目的覆盖写必须留档，且名字带根名前缀。

    为什么强调前缀：data/backup/ 是扁平目录，不同项目的同名文件（README.md）
    不加前缀就会互相覆盖，回滚时拿到别人的版本。
    """
    extra = tmp_path / "extproj"
    extra.mkdir()
    target = extra / "README.md"
    target.write_text("v1", encoding="utf-8")
    builtin.set_workspace_roots([str(extra)])

    backup = builtin._backup_before_overwrite(target)           # noqa: SLF001
    assert backup is not None
    # 新前缀格式 =「根名_短哈希__相对路径」：extproj_<hash>__README.md...
    name = Path(backup).name
    assert name.startswith("extproj_") and "__README" in name
    assert Path(backup).read_text(encoding="utf-8") == "v1"
    Path(backup).unlink(missing_ok=True)


def test_backup_prefix_unique_for_same_named_roots(tmp_path):
    """两个不同目录若恰好同名（都叫 proj），备份名必须靠哈希区分，不能互相覆盖。

    这是修复「同名根互相覆盖」回归的核心断言：修复前两份 README.md 的留档
    会生成同一个 basename，回滚时拿错版本。
    """
    a = tmp_path / "a" / "proj"
    b = tmp_path / "b" / "proj"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "README.md").write_text("来自 a", encoding="utf-8")
    (b / "README.md").write_text("来自 b", encoding="utf-8")
    builtin.set_workspace_roots([str(a), str(b)])

    bak_a = builtin._backup_before_overwrite(a / "README.md")    # noqa: SLF001
    bak_b = builtin._backup_before_overwrite(b / "README.md")    # noqa: SLF001
    assert bak_a and bak_b
    assert Path(bak_a).name != Path(bak_b).name
    assert Path(bak_a).read_text(encoding="utf-8") == "来自 a"
    assert Path(bak_b).read_text(encoding="utf-8") == "来自 b"
    Path(bak_a).unlink(missing_ok=True)
    Path(bak_b).unlink(missing_ok=True)


def test_workspace_rejects_filesystem_root(tmp_path):
    """把盘符根/文件根配成工作区会交出整机，必须拒绝（防呆）。"""
    root = tmp_path
    while root.parent != root:        # 逐级向上直到文件系统根（Win 上是 C:\，Linux 上是 /）
        root = root.parent
    roots = builtin.set_workspace_roots([str(root)])
    assert builtin.ROOT_RESOLVED in roots
    assert root.resolve() not in roots
    # 根目录下的越界尝试仍被拒（确认没有因为「配了根」而放开整机）
    assert builtin._resolve(str(root / "Windows" / "System32" / "config")) is None  # noqa: SLF001


def test_registry_reads_external_workspace(tmp_path):
    """走真实工具入口端到端验证（与 bridge 调用同一条路径）。"""
    extra = tmp_path / "extproj"
    extra.mkdir()
    (extra / "note.txt").write_text("外部内容", encoding="utf-8")

    reg = builtin.build_builtin_tools(_FakeCfg([str(extra)]))
    assert "外部内容" in reg.execute("read_file", {"path": str(extra / "note.txt")})
    assert "note.txt" in reg.execute("list_dir", {"path": str(extra)})
    # 越界路径必须仍被拒
    assert "超出项目根目录" in reg.execute("read_file", {"path": "C:/Windows/win.ini"})
