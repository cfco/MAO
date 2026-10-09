"""变异测试工具：逐项把修复改回缺陷写法，确认对应用例真的失败。

用法（不参与 pytest 收集，文件名不以 test_ 开头）：
    uv run python tests/mutation_check.py        # 用当前解释器，Windows/Linux/CI 同一命令
    ./.venv/Scripts/python.exe tests/mutation_check.py   # 老的写法仍可用

为什么必须做：回归用例如果断言写歪了，无论代码正确与否都会通过，等于没测。
本工具对每处修复做"反向变异"——撤销修复 → 期望相关用例失败 → 还原源码。
任何一条"变异后仍然通过"的用例都是恒真用例，必须重写。

注意：MUTATIONS 里的锚点是**修复后的源码片段**，代码再演进后锚点可能失配，
失配会打印 SKIP 提示（不会误改源码）。**SKIP 也算失败**（退出码非 0）：锚点失配
意味着这条回归保护已经不再被校验，放任它会悄悄变成"没测"。新增修复时按同样格式
追加条目即可。重构（抽公共函数/合并重复口径）后务必回来把指向旧写法的锚点改指新位置。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 用跑本脚本的解释器起 pytest：写死 .venv/Scripts/python.exe 是 Windows 路径，
# 在 CI（Linux，venv 里是 bin/python）上会直接 FileNotFoundError。
PY = sys.executable
TESTFILE = "tests/test_audit_fixes.py"

MUTATIONS: list[tuple[str, str, str, str, str]] = [
    (
        "M1 健康档案不再共享",
        "agent/core/health.py",
        """    return shared_by_path(
        _shared_instances,
        _shared_lock,
        key,
        lambda: ModelHealth(key, registry_path, retire_days),
    )""",
        """    return ModelHealth(key)  # mutated: 不共享，各池各持一本账""",
        f"{TESTFILE}::test_get_health_returns_same_instance_per_path",
    ),
    (
        "M2 健康档案不共享（覆盖写盘）",
        "agent/core/health.py",
        """    return shared_by_path(
        _shared_instances,
        _shared_lock,
        key,
        lambda: ModelHealth(key, registry_path, retire_days),
    )""",
        """    return ModelHealth(key)  # mutated: 不共享，各池各持一本账""",
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
        "        return [w for w in dict.fromkeys(workers or self.pick()) if w in self.profiles]",
        "        return [w for w in dict.fromkeys(workers or self.names())  # mutated\n"
        "                if w in self.profiles]",
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
        "M17 档案落盘 IO 拉回锁内",
        "agent/core/ledger.py",
        """        with self._write_lock:
            with self._lock:
                text = self._snapshot_locked()
            try:
                atomic_write_text(self.store_path, text)
            except OSError:
                pass  # 档案写不进去只损失持久化，内存账本照常工作""",
        """        with self._write_lock, self._lock:  # mutated: 落盘 IO 回到记账锁内
            text = self._snapshot_locked()
            try:
                atomic_write_text(self.store_path, text)
            except OSError:
                pass""",
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
    (
        "M20 同站回退被关闭",
        "agent/core/orchestrator.py",
        "        self._fallback_models = self.cfg.fallback_models",
        "        self._fallback_models = 0  # mutated: 关闭回退",
        "tests/test_core.py::test_fallback_to_peer_on_retryable_failure",
    ),
    (
        "M21 批次死站记忆不再抑制同站回退",
        "agent/core/orchestrator.py",
        "                    fb = self._fallback_target(p, tried, memo)",
        "                    fb = self._fallback_target(p, tried, None)  # mutated: 忽略批次死站记忆",
        "tests/test_core.py::test_station_memo_suppresses_cross_worker_fallback",
    ),
    (
        "M22 跨站回退被关闭（默认应开）",
        "agent/core/orchestrator.py",
        "        self._fallback_cross_station = self.cfg.fallback_cross_station",
        "        self._fallback_cross_station = False  # mutated: 关闭跨站恢复",
        "tests/test_core.py::test_fallback_cross_station_recovers",
    ),
    (
        "M23 延迟档案不再共享",
        "agent/core/latency.py",
        "    return shared_by_path(_shared_instances, _shared_lock, key, lambda: LatencyStore(key))",
        """    return LatencyStore(key)  # mutated: 不共享，各池各写各的""",
        "tests/test_latency.py::test_get_latency_shares_instance_per_path",
    ),
    (
        "M24 派工入口不做延迟裁剪",
        "agent/core/orchestrator.py",
        """        if limit is None:
            live = self._latency_window(live)""",
        """        if False:  # mutated: 关闭延迟裁剪
            live = self._latency_window(live)""",
        "tests/test_latency.py::test_pick_drops_slowest_share_and_sorts_by_latency",
    ),
    (
        "M25 延迟窗口不再裁剪（只排序）",
        "agent/core/latency.py",
        "        drop = int(len(ordered) * (1.0 - self._latency_keep_ratio))",
        "        drop = 0  # mutated: 不裁掉高延迟的",
        "tests/test_latency.py::test_pick_small_pool_keeps_two_or_three",
    ),
    (
        "M26 冷启动保护被去掉（无样本也裁剪）",
        "agent/core/latency.py",
        """        if len(known) < min(self._latency_min_samples, len(live)):
            return live""",
        """        if False:  # mutated: 证据不足也动刀
            return live""",
        "tests/test_latency.py::test_pick_without_enough_latency_data_does_not_prune",
    ),
    (
        "M27 探测只覆盖可用节点",
        "agent/core/latency.py",
        "            self._prober.maybe_start(list(self.profiles))",
        "            self._prober.maybe_start([w for w in self.profiles"
        " if self._skip_reason(w) is None])  # mutated: 漏掉不可用节点",
        "tests/test_latency.py::test_touch_latency_probe_covers_unavailable_nodes",
    ),
    (
        "M28 无视自动探测开关（关不掉）",
        "agent/core/latency.py",
        """        if not self._latency_auto_probe:
            return""",
        """        if False:  # mutated: 关不掉自动探测
            return""",
        "tests/test_latency.py::test_probe_disabled_by_config_does_not_probe",
    ),
    (
        "M29 health 快照不再带延迟",
        "agent/core/orchestrator.py",
        '                "latency_ms": lat.get(name),',
        '                "latency_ms": None,  # mutated: 外部主看不到延迟',
        "tests/test_latency.py::test_bridge_latency_command_reports_and_refreshes",
    ),
    # ---------------- 第四轮审计（2026-10-09）新增 ----------------
    (
        "M30 POSIX shell 家族不进内联扫描",
        "agent/tools/shell_safety.py",
        '    "sh",\n    "bash",\n    "zsh",\n    "dash",\n    "ksh",\n}',
        "}",
        "tests/test_core.py::test_posix_shell_inline_guard_blocks_bypass",
    ),
    (
        "M31 shell 短选项簇 -lc 不认",
        "agent/tools/shell_safety.py",
        r"""_SHELL_CODE_FLAG = re.compile(r"-[a-z]{0,3}c\Z")""",
        r"""_SHELL_CODE_FLAG = re.compile(r"-c\Z")  # mutated: 只认光秃秃的 -c""",
        "tests/test_core.py::test_posix_shell_inline_guard_blocks_bypass",
    ),
    (
        "M32 错误文案不再脱敏",
        "agent/core/redact.py",
        """    for pat, repl in _SECRET_PATTERNS:
        text = pat.sub(repl, text)""",
        "    pass  # mutated: 原文透传，凭证会跟着进 stdout 协议行",
        "tests/test_audit4_fixes.py::test_llm_fallback_error_does_not_leak_key",
    ),
    (
        "M33 取快照挪到写锁之外",
        "agent/core/ledger.py",
        """        with self._write_lock:
            with self._lock:
                text = self._snapshot_locked()""",
        """        with self._lock:  # mutated: 快照在写锁外取，写盘顺序可能反于快照顺序
            text = self._snapshot_locked()
        with self._write_lock:""",
        "tests/test_audit4_fixes.py::test_snapshot_is_taken_inside_the_write_lock[health]",
    ),
    (
        "M34 dotenv 注释剥离抢在引号处理之前",
        "agent/config.py",
        """            val = val.strip()
            if val[:1] in ('"', "'"):""",
        """            val = val.split(" #", 1)[0].strip().strip('"').strip("'")
            if False:  # mutated: 旧顺序，引号内的 # 被当注释切掉""",
        "tests/test_audit4_fixes.py::test_dotenv_keeps_hash_inside_quotes",
    ),
    (
        "M36 冷却时限改用墙上时钟",
        "agent/core/orchestrator.py",
        "        return time.monotonic() < until",
        "        return time.time() < until  # mutated: NTP 校时即可提前解除冷却",
        "tests/test_audit4_fixes.py::test_cooldown_is_immune_to_wall_clock_jump",
    ),
    (
        "M37 客户端缓存压在池记账锁里",
        "agent/core/orchestrator.py",
        """        with self._clients_lock:
            client = self._clients.get(p.name)""",
        """        with self._lock:  # mutated: 建 httpx/SSL 期间把纯内存记账串行化
            client = self._clients.get(p.name)""",
        "tests/test_audit4_fixes.py::test_client_construction_does_not_hold_pool_lock",
    ),
    (
        "M38 模型名按最后一个 @ 硬切",
        "agent/model_registry.py",
        "    bare = name.strip() if sep and _SIZE_SUFFIX.fullmatch(suffix.strip()) else body.strip()",
        '    bare = body.rpartition("@")[0].strip()  # mutated: 不带 @ 的条目被切成空串',
        "tests/test_audit4_fixes.py::test_retire_model_reports_three_states",
    ),
    (
        "M39 强制探测可与自动轮叠跑",
        "agent/core/latency.py",
        """        with self._round_lock:
            with self._lock:""",
        """        if True:  # mutated: 不经轮锁，两轮同时打网络
            with self._lock:""",
        "tests/test_latency.py::test_forced_probe_never_overlaps_auto_round",
    ),
    (
        "M40 线程起不来不回滚探测名额",
        "agent/core/latency.py",
        """            self._release_claim()
            return False""",
        "            return False  # mutated: 名额悬空，自动探测永久停摆",
        "tests/test_latency.py::test_maybe_start_rolls_back_claim_when_thread_fails",
    ),
    (
        "M41 技能扫描不容错（单文件炸启动）",
        "agent/skills_manager.py",
        """            except (OSError, ValueError) as e:
                print(f"[技能警告] 跳过无法读取的 {skill_md}：{type(e).__name__}", file=sys.stderr)
                continue""",
        "            except FileNotFoundError:  # mutated: 其它读取失败逃出整轮扫描\n"
        "                continue",
        "tests/test_core.py::test_skill_scan_survives_bad_encoding",
    ),
    (
        "M42 write_file 不前置判断目录",
        "agent/tools/builtin.py",
        "        if p.is_dir():\n            # 不前置判断的话",
        "        if p.is_dir() and False:  # mutated: 让 IsADirectoryError 逃到上层兜底\n"
        "            # 不前置判断的话",
        "tests/test_audit4_fixes.py::test_write_and_append_reject_directory",
    ),
    (
        "M43 并行度只看总上限、不折算站数",
        "agent/core/station_gate.py",
        "        return max(1, min(cap, per_station * max(1, stations)))",
        "        return cap  # mutated: 同站串行不再影响批次口径",
        "tests/test_audit4_fixes.py::test_station_parallel_cap_is_one_formula",
    ),
]


def run_pytest(node: str) -> bool:
    """返回是否通过。"""
    proc = subprocess.run(
        [PY, "-m", "pytest", "-q", "-x", node],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=600,
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
    rows.append(
        (
            label,
            "OK" if not passed else "TAUTOLOGY",
            "变异后失败=用例有效" if not passed else "变异后仍通过=恒真用例",
        )
    )

print("=" * 78)
print(f"{'变异':<26}{'结果':<12}说明")
print("-" * 78)
for label, verdict, note in rows:
    print(f"{label:<26}{verdict:<12}{note}")
bad = [r for r in rows if r[1] != "OK"]
print("=" * 78)
print("全部用例均为有效断言（去掉修复即失败）" if not bad else f"存在可疑用例：{bad}")
sys.exit(1 if bad else 0)
