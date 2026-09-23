"""变异测试工具：逐项把修复改回缺陷写法，确认对应用例真的失败。

用法（不参与 pytest 收集，文件名不以 test_ 开头）：
    ./.venv/Scripts/python.exe tests/mutation_check.py

为什么必须做：回归用例如果断言写歪了，无论代码正确与否都会通过，等于没测。
本工具对每处修复做"反向变异"——撤销修复 → 期望相关用例失败 → 还原源码。
任何一条"变异后仍然通过"的用例都是恒真用例，必须重写。

注意：MUTATIONS 里的锚点是**修复后的源码片段**，代码再演进后锚点可能失配，
失配会打印 SKIP 提示（不会误改源码）。新增修复时按同样格式追加条目即可。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
TESTFILE = "tests/test_audit_fixes.py"

MUTATIONS: list[tuple[str, str, str, str, str]] = [
    (
        "M1 健康档案不再共享",
        "agent/core/health.py",
        """    key = Path(store_path)
    with _shared_lock:
        inst = _shared_instances.get(key)
        if inst is None:
            inst = ModelHealth(key)
            _shared_instances[key] = inst
        return inst""",
        """    return ModelHealth(Path(store_path))""",
        f"{TESTFILE}::test_get_health_returns_same_instance_per_path",
    ),
    (
        "M2 健康档案不共享（覆盖写盘）",
        "agent/core/health.py",
        """    key = Path(store_path)
    with _shared_lock:
        inst = _shared_instances.get(key)
        if inst is None:
            inst = ModelHealth(key)
            _shared_instances[key] = inst
        return inst""",
        """    return ModelHealth(Path(store_path))""",
        f"{TESTFILE}::test_shared_health_does_not_lose_each_others_records",
    ),
    (
        "M3 pick() 不再限制人数",
        "agent/core/orchestrator.py",
        "        return window[:cap]",
        "        return window",
        f"{TESTFILE}::test_pick_caps_participants_and_keeps_order",
    ),
    (
        "M4 限时改回固定 300s",
        "agent/core/parallelism.py",
        "        return batches * per_call + 30.0",
        "        return 300.0",
        f"{TESTFILE}::test_effective_timeout_scales_with_batches_and_respects_explicit",
    ),
    (
        "M5 MCP 连接不再共享",
        "agent/tools/mcp_client.py",
        "                _groups[key] = group",
        "                _groups[key + '!'] = group",
        f"{TESTFILE}::test_mcp_group_shared_and_refcounted",
    ),
    (
        "M8 计票不过滤非成功票",
        "agent/core/voting.py",
        """            if res is None or res["status"] != ST_OK:
                continue""",
        """            if res is None:
                continue""",
        f"{TESTFILE}::test_ballot_ignores_error_text",
    ),
    (
        "M10 vote 缺省名单改回全池",
        "agent/core/voting.py",
        "        pool = [w for w in dict.fromkeys(workers or self.pick()) if w in self.profiles]",
        "        pool = [w for w in dict.fromkeys(workers or self.names()) if w in self.profiles]",
        f"{TESTFILE}::test_default_fanout_is_bounded",
    ),
    (
        "M12 投票分母含无效票",
        "agent/core/voting.py",
        """        total = len(valid)
        invalid = len(votes) - total""",
        """        total = len(votes)
        invalid = 0""",
        "tests/test_audit2_fixes.py::test_ballot_denominator_excludes_invalid_votes",
    ),
    (
        "M13 数值配置裸转换",
        "agent/config.py",
        """        try:
            out = int(value)
        except (TypeError, ValueError):
            _warn_bad_value(key, value, default)
            out = default
        return out if minimum is None else max(minimum, out)""",
        """        return int(value)""",
        "tests/test_audit2_fixes.py::test_bad_numeric_config_falls_back",
    ),
    (
        "M15 write_file 覆盖不留档",
        "agent/tools/builtin.py",
        "        backup = _backup_before_overwrite(p)  # 覆盖前留档，保证可回滚",
        "        backup = None  # mutated",
        "tests/test_audit2_fixes.py::test_write_file_backs_up_existing_file",
    ),
    (
        "M16 注入检查不屏蔽引号",
        "agent/tools/shell_safety.py",
        "    scan = _mask_quoted(stripped)",
        "    scan = stripped  # mutated",
        "tests/test_audit2_fixes.py::test_injection_check_ignores_quoted_literals",
    ),
    (
        "M17 健康档案写盘拉回锁内",
        "agent/core/health.py",
        """        tmp: Path | None = None
        with self._write_lock:""",
        """        tmp: Path | None = None
        with self._lock, self._write_lock:  # mutated""",
        "tests/test_audit2_fixes.py::test_persist_does_not_block_quarantine_queries",
    ),
    (
        "M18 408/425 不再重试",
        "agent/core/llm.py",
        "            if code >= 500 or code in _RETRYABLE_STATUS:",
        "            if code >= 500:  # mutated",
        "tests/test_audit2_fixes.py::test_http_status_retry_classification",
    ),
    (
        "M19 解释器内联代码不再扫描",
        "agent/tools/shell_safety.py",
        "            return interp, _strip_wrapping_quotes(tokens[i + 1])",
        '            return interp, ""  # mutated',
        "tests/test_core.py::test_interpreter_inline_code_guard_blocks_bypass",
    ),
]


def run_pytest(node: str) -> bool:
    """返回是否通过。"""
    proc = subprocess.run(
        [PY, "-m", "pytest", "-q", "-x", node],
        cwd=str(ROOT), capture_output=True, text=True, timeout=600,
    )
    return proc.returncode == 0


rows: list[tuple[str, str, str]] = []
for label, rel, old, new, node in MUTATIONS:
    path = ROOT / rel
    original = path.read_text(encoding="utf-8")
    if old not in original:
        rows.append((label, "SKIP", "锚点未找到（源码已变）"))
        continue
    try:
        path.write_text(original.replace(old, new, 1), encoding="utf-8")
        passed = run_pytest(node)
    finally:
        path.write_text(original, encoding="utf-8")
    rows.append((label, "OK" if not passed else "TAUTOLOGY", "变异后失败=用例有效" if not passed else "变异后仍通过=恒真用例"))

print("=" * 78)
print(f"{'变异':<26}{'结果':<12}说明")
print("-" * 78)
for label, verdict, note in rows:
    print(f"{label:<26}{verdict:<12}{note}")
bad = [r for r in rows if r[1] != "OK"]
print("=" * 78)
print("全部用例均为有效断言（去掉修复即失败）" if not bad else f"存在可疑用例：{bad}")
sys.exit(1 if bad else 0)
