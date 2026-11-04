"""变异测试工具：逐项把修复改回缺陷写法，确认对应用例真的失败。

用法（不参与 pytest 收集，文件名不以 test_ 开头）：
    uv run python tests/mutation_check.py        # 用当前解释器，Windows/Linux/CI 同一命令
    ./.venv/bin/python tests/mutation_check.py   # Linux venv 布局（Windows 为 .venv/Scripts/python.exe）

为什么必须做：回归用例如果断言写歪了，无论代码正确与否都会通过，等于没测。
本工具对每处修复做"反向变异"——撤销修复 → 期望相关用例失败 → 还原源码。
任何一条"变异后仍然通过"的用例都是恒真用例，必须重写。

注意：MUTATIONS 里的锚点是**修复后的源码片段**，代码再演进后锚点可能失配，
失配会打印 SKIP 提示（不会误改源码）。**SKIP 也算失败**（退出码非 0）：锚点失配
意味着这条回归保护已经不再被校验，放任它会悄悄变成"没测"。新增修复时按同样格式
追加条目即可。重构（抽公共函数/合并重复口径）后务必回来把指向旧写法的锚点改指新位置。
"""

from __future__ import annotations

import ast
import os
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
        "        return batches * chain * worst + 30.0",
        "        return 300.0  # mutated: 固定限时会截断尾部批次",
        f"{TESTFILE}::test_effective_timeout_scales_with_batches_and_respects_explicit",
    ),
    (
        "M5 MCP 连接不再共享",
        "agent/tools/mcp_group.py",
        "                _groups[key] = fresh",
        "                _groups[key + '!'] = fresh",
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
        "agent/config_util.py",
        "def as_int(value: Any, key: str, default: int, minimum: int | None = None) -> int:",
        'def as_int(value, key="k", default=0, minimum=None):\n'
        "    return int(value)  # mutated: 裸转换，类型错/inf 直接崩",
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
        """                    text = self._snapshot_locked()
                if raw is None or self._read_disk_text() == raw:
                    break
                disk, raw = self._disk_for_merge()  # 被对端改过：拿新的一份重合并
            try:
                atomic_write_text(self.store_path, text)""",
        """                    text = self._snapshot_locked()
                    atomic_write_text(self.store_path, text)  # mutated: 落盘 IO 回到记账锁内
                if raw is None or self._read_disk_text() == raw:
                    break
                disk, raw = self._disk_for_merge()
            try:
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
        "agent/core/dispatch.py",
        """                    if err.retryable and fallbacks_left > 0:
                        fb = self._fallback_target(p, tried, memo)""",
        """                    if err.retryable and fallbacks_left > 0:
                        fb = self._fallback_target(p, tried, None)  # mutated: 忽略批次死站记忆""",
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
        "agent/core/latency_store.py",
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
        "        drop = math.floor(len(ordered) * (1.0 - self._latency_keep_ratio) + 1e-9)",
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
    # 注：原 M30「POSIX shell 家族不进内联扫描」已撤下锚点（2026-10-29 第八轮）：
    # `sh -c "rm -rf x"` 现在是**双层**拦截——_INTERP_INLINE 的内联正文扫描 + _RUNNER_HEADS
    # 的包装展开，单点变异任意一层另一层都仍拦住，留着只会记成 TAUTOLOGY 误报恒真。
    # 该用例（test_posix_shell_inline_guard_blocks_bypass）的载荷性由 M31（-lc 短选项簇）
    # 与 M82（Runner 正文不展开）指向同一/同源节点证明。编号按仓库惯例留空号不重排。
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
            disk, raw = self._disk_for_merge()  # IO 留在 _lock 之外""",
        """        with threading.Lock():  # mutated: 每次现造一把锁＝写盘之间不再串行，快照顺序可能反于写盘顺序
            disk, raw = self._disk_for_merge()""",
        "tests/test_audit4_fixes.py::test_snapshot_is_taken_inside_the_write_lock[health]",
    ),
    (
        "M34 dotenv 注释剥离抢在引号处理之前",
        "agent/config_load.py",
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
        "    bare = name.strip() if sep and WINDOW_SIZE_SUFFIX.fullmatch(suffix.strip()) else body.strip()",
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
    (
        "M44 发现对账保留 # 前缀（屏蔽模型被复活）",
        "agent/core/node_discovery.py",
        '    registry_active = {_bare(i).lstrip("#") for i in items if not i.startswith("#")}\n'
        '    registry_all = {_bare(i).lstrip("#") for i in items}',
        '    registry_active = {_bare(i) for i in items if not i.startswith("#")}\n'
        "    registry_all = {_bare(i) for i in items}",
        "tests/test_node_discovery.py::test_reconcile_ignores_manually_blocked_names",
    ),
    (
        "M45 registry 行内注释不剥离（末个模型不可下线）",
        "agent/model_registry.py",
        '            value, comment = _split_value_comment(line.partition("=")[2].strip())',
        '            value, comment = (line.partition("=")[2].strip(), "")',
        "tests/test_core.py::test_registry_inline_comment_does_not_block_last_model_retire",
    ),
    (
        "M46 账本写盘不并对方进程记录（互相覆盖）",
        "agent/core/ledger.py",
        "                with self._lock:\n"
        "                    if disk is not None:\n"
        "                        self._merge_disk_locked(disk)\n"
        "                    text = self._snapshot_locked()\n",
        "                with self._lock:\n"
        "                    text = self._snapshot_locked()  # mutated: 不并对端记录，整篇覆盖\n",
        "tests/test_health.py::test_write_merges_records_from_another_process",
    ),
    (
        "M47 成功派工不登记运行日",
        "agent/core/orchestrator.py",
        "        if succeeded:\n"
        "            self._note_run_day()  # 登记运行日同样是落盘 IO，放在池锁外\n"
        "            return",
        "        if succeeded:\n            return  # mutated: 成功日不进 active_dates",
        "tests/test_core.py::test_successful_dispatch_registers_run_day",
    ),
    (
        # 锚点必须对准 else 分支（shell=True）那次 Popen：用例命令 "set | find /c KEY"
        # 含管道，_try_split_args 直接返回 None 走 else，参数模式/FileNotFoundError
        # 回退那两处被变异不影响本用例。缩进别抄错：另两处的实参缩进多一级。
        "M48 子进程继承全量 env（key 可被 set 抄走）",
        "agent/tools/builtin.py",
        "                proc = subprocess.Popen(\n"
        "                    command,\n"
        "                    shell=True,\n"
        "                    cwd=cwd,\n"
        "                    stdout=subprocess.PIPE,\n"
        "                    stderr=subprocess.PIPE,\n"
        "                    env=_child_env(),\n"
        "                    start_new_session=True,\n"
        "                )",
        "                proc = subprocess.Popen(\n"
        "                    command,\n"
        "                    shell=True,\n"
        "                    cwd=cwd,\n"
        "                    stdout=subprocess.PIPE,\n"
        "                    stderr=subprocess.PIPE,\n"
        "                    start_new_session=True,\n"
        "                )",
        "tests/test_core.py::test_run_shell_child_env_has_no_secret_vars",
    ),
    (
        "M49 read_file 不拒绝密钥文件",
        "agent/tools/builtin.py",
        """        if _is_secret_file(p):
            # 允许根之内不等于可以外发：`.env` 就躺在项目根里，read_file 原先把它整份读出，""",
        """        if False:  # mutated: 允许根之内即可读出 .env
            # 允许根之内不等于可以外发：`.env` 就躺在项目根里，read_file 原先把它整份读出，""",
        "tests/test_core.py::test_read_file_refuses_env_secret",
    ),
    (
        "M50 脱敏漏掉下划线后缀的 key 名",
        "agent/core/redact.py",
        "    (_SECRET_KV, _mask_secret_kv),",
        "    (_SECRET_KV, lambda m: m.group(0)),  # mutated: KV 一律不抹",
        "tests/test_core.py::test_redact_masks_underscore_suffixed_secret_names",
    ),
    (
        "M51 对外池清单读静态配置（热同步后失真）",
        "agent/bridge.py",
        '            return {"ok": True, "agents": [p.brief() for p in self.workers.profiles.values()]}',
        '            return {"ok": True, "agents": [p.brief() for p in self.cfg.agent_profiles]}',
        "tests/test_audit5_fixes.py::test_list_agents_and_ping_report_the_live_pool",
    ),
    (
        "M52 协议层不透传 timeout",
        "agent/bridge.py",
        '                req.get("system"),\n                timeout=_norm_timeout(req.get("timeout")),',
        '                req.get("system"),',
        "tests/test_audit5_fixes.py::test_ask_many_passes_timeout_and_tags_to_the_kernel",
    ),
    (
        "M53 interval=0 连手动发现一起废掉",
        "agent/core/node_discovery.py",
        "        return bool(self.cfg.free_node_discovery)",
        "        return bool(self.cfg.free_node_discovery) and self._interval > 0",
        "tests/test_node_discovery.py::test_manual_refresh_still_works_when_interval_is_zero",
    ),
    (
        "M54 起线程失败不回滚周期起算点",
        "agent/core/node_discovery.py",
        "                self._running = False\n                self._last_start = self._last_finish",
        "                self._running = False",
        "tests/test_node_discovery.py::test_maybe_start_rolls_back_the_schedule_when_thread_fails",
    ),
    (
        "M55 已关闭的 MCP 组被直接复用",
        "agent/tools/mcp_group.py",
        "        with self._lock:\n"
        "            if self._closed:\n"
        "                return False\n"
        "            self._refs += 1\n"
        "            return True",
        "        with self._lock:\n            self._refs += 1\n            return True",
        "tests/test_audit5_fixes.py::test_acquire_group_rebuilds_a_closed_group_instead_of_raising",
    ),
    (
        "M56 叠加层读取忽略 collaboration.free_nodes_file",
        "agent/config_load.py",
        '    overlay_path = _free_nodes_path(data.get("collaboration") or {})',
        "    overlay_path = _free_nodes_path({})",
        "tests/test_node_discovery.py::test_load_config_applies_overlay_from_the_configured_path",
    ),
    (
        # 注：换行防御是"入口拦截 + 切段正则"两层，单点变异另一层仍然拦得住，
        # 所以这里不给它单独条目（否则只能记成 TAUTOLOGY 误报恒真）；
        # 该 parametrize 用例的载荷性由 M58 / M59 / M60 指向同一节点来证明。
        "M57 首词不剥 cmd 的 ^ 转义",
        "agent/tools/shell_rules.py",
        '    low = _strip_wrapping_quotes(raw).replace("^", "")',
        "    low = _strip_wrapping_quotes(raw)  # mutated: de^l 放行",
        "tests/test_audit5_fixes.py::test_shell_safety_blocks_the_six_bypasses",
    ),
    (
        "M58 EncodedCommand 检查未接入",
        "agent/tools/shell_safety.py",
        "    ps_enc = _check_powershell_encoding(stripped)",
        "    ps_enc = None  # mutated: 只按首词判 powershell，串联/引号包裹都绕过",
        "tests/test_audit5_fixes.py::test_shell_safety_blocks_the_six_bypasses",
    ),
    (
        "M59 重定向目标不拦环境变量展开",
        "agent/tools/shell_rules.py",
        '        if "%" in target or target.startswith(("$", "~")):',
        "        if False:  # mutated: %USERPROFILE% 之类直接放行",
        "tests/test_audit5_fixes.py::test_shell_safety_blocks_the_six_bypasses",
    ),
    (
        "M60 技能脚本不看退出码",
        "agent/skills_manager.py",
        "        if proc.returncode != 0:",
        "        if False:  # mutated: 非 0 退出也当成功",
        "tests/test_audit5_fixes.py::test_run_script_marks_nonzero_exit_as_failure",
    ),
    (
        "M61 被丢掉的陌生投票人不点名",
        "agent/core/voting.py",
        """        unknown = [
            w for w in dict.fromkeys(str(x) for x in (workers or [])) if w not in self.profiles
        ]""",
        "        unknown = []  # mutated: 静默丢掉不存在的名字",
        "tests/test_audit5_fixes.py::test_vote_names_the_workers_it_had_to_drop",
    ),
    (
        "M62 整体限时不算重试与退避",
        "agent/core/parallelism.py",
        "        worst = worst_case_per_request(self._llm_timeout, self._max_retries)",
        "        worst = self._llm_timeout  # mutated: 只按单次 timeout 折算，尾部批次被误判未完成",
        "tests/test_audit_fixes.py::test_effective_timeout_scales_with_batches_and_respects_explicit",
    ),
    (
        "M63 派工线程数随点名人头数线性创建",
        "agent/core/parallelism.py",
        "        slots = min(len(workers), max(1, self.cfg.max_workers))",
        "        slots = len(workers)  # mutated: 回退到一人一线程",
        "tests/test_audit5_fixes.py::test_dispatch_thread_count_follows_concurrency_not_fanout",
    ),
    (
        "M64 门又包住整条重试链（退避在门内睡）",
        "agent/core/station_gate.py",
        "                return client.chat_once(messages), None, (time.monotonic() - t0) * 1000.0",
        "                return client.chat(messages), None, (time.monotonic() - t0) * 1000.0",
        "tests/test_audit5_fixes.py::test_backoff_sleep_happens_outside_the_station_gate",
    ),
    (
        "M65 门外重试环不再重试",
        "agent/core/dispatch.py",
        "            if attempt >= attempts - 1:",
        "            if True:  # mutated: 一次就交回上层，等于派工侧关掉重试",
        "tests/test_audit5_fixes.py::test_backoff_sleep_happens_outside_the_station_gate",
    ),
    (
        "M66 list_dir 条目中途 stat 失败仍炸整次",
        "agent/tools/builtin.py",
        '                    entries.append(f"?     {item.name}（读取失败）")\n                    continue',
        "                    raise  # mutated: 逃到注册表兜底，整次列目录失败",
        "tests/test_audit5_fixes.py::test_list_dir_survives_an_entry_vanishing_mid_scan",
    ),
    (
        "M67 留档名不带路径哈希（深层与平铺撞名）",
        "agent/tools/backup.py",
        '        digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]',
        '        digest = "00000000"  # mutated: 回到 a/b.md 与 a__b.md 同名的旧行为',
        "tests/test_audit5_fixes.py::test_backup_names_do_not_collide_across_shapes",
    ),
    (
        "M68 显式 timeout=0 又当成限时 0 秒",
        "agent/core/parallelism.py",
        "        if timeout is not None and float(timeout) > 0:",
        "        if timeout is not None:",
        "tests/test_audit5_fixes.py::test_nonpositive_explicit_timeout_falls_back_to_adaptive",
    ),
    (
        "M69 兜底未知异常又进当日隔离",
        "agent/core/dispatch.py",
        """                    self._record_result(
                        p.name, ok=False, fatal=not err.retryable and err.error_type != ERR_OTHER
                    )""",
        """                    self._record_result(
                        p.name, ok=False, fatal=not err.retryable
                    )  # mutated: 未知异常也当日隔离""",
        "tests/test_audit6_fixes.py::test_unknown_exception_only_cools_not_quarantines",
    ),
    (
        "M70 被移出工人的连接池不回收",
        "agent/core/orchestrator.py",
        """        with self._clients_lock:
            for name in removed:
                for cache in (self._clients, self._probe_clients):
                    client = cache.pop(name, None)
                    if client is not None:
                        retired.append(client)""",
        "        retired: list[LLMClient] = []  # mutated: 不回收，连接池挂到进程退出",
        "tests/test_audit6_fixes.py::test_removed_worker_clients_are_reclaimed",
    ),
    (
        "M71 子进程环境只认分段式密钥名（漏 MYKEY/GHTOKEN）",
        "agent/tools/shell_rules.py",
        """        if name.rsplit("_", 1)[-1] in SECRET_NAME_TAILS or any(
            word in name for word in SECRET_NAME_SUBSTRINGS
        ):""",
        '        if name.rsplit("_", 1)[-1] in SECRET_NAME_TAILS:  # mutated: 只看分段后缀',
        "tests/test_audit6_fixes.py::test_child_env_strips_compact_secret_names",
    ),
    (
        "M72 探测遇参数不支持不降级（整轮样本丢光）",
        "agent/core/latency.py",
        """        if self._latency_probe_max_tokens <= 0 or station in self._probe_no_max_tokens:
            return False
        if err.retryable:
            return False
        self._probe_no_max_tokens.add(station)
        return True""",
        "        return False  # mutated: 不降级，不兼容的网关会让整轮样本丢光",
        "tests/test_audit6_fixes.py::test_probe_degrades_to_no_max_tokens_per_station",
    ),
    (
        "M73 投票预览长度又写死 80",
        "agent/core/voting.py",
        "        preview_chars = self.cfg.vote_preview_chars",
        "        preview_chars = 80  # mutated: 预览长度写死",
        "tests/test_audit6_fixes.py::test_vote_preview_length_is_configurable",
    ),
    (
        "M74 MCP 工具调用超时又写死 300",
        "agent/tools/mcp_client.py",
        "            result = fut.result(timeout=self._call_timeout)",
        "            result = fut.result(timeout=300)  # mutated: 超时写死",
        "tests/test_audit6_fixes.py::test_mcp_tool_timeout_is_configurable",
    ),
    # ---------------- 第七轮审计（2026-10-22）：Windows/Ubuntu 双平台兼容 ----------------
    (
        "M75 POSIX 顶层词表又只剩 Windows 集合",
        "agent/tools/shell_safety.py",
        """    # 只是多认几个 Git-Bash/WSL 里可能出现的词，正常开发命令不受影响）。
    for h in heads:
        if _is_dangerous_word(h):
            return f"命令拦截：检测到危险命令 `{h}`\"""",
        """    # 只是多认几个 Git-Bash/WSL 里可能出现的词，正常开发命令不受影响）。
    for h in heads:
        if h in _CMD_DANGEROUS_SET:  # mutated: 顶层又只看 Windows 词表
            return f"命令拦截：检测到危险命令 `{h}`\"""",
        "tests/test_platform_compat.py::test_posix_dangerous_commands_blocked_at_top_level",
    ),
    (
        "M76 超时不杀进程树（孙进程残留）",
        "agent/tools/builtin.py",
        "                _kill_process_tree(proc)",
        "                pass  # mutated: 超时后不杀树，孙进程留成孤儿",
        "tests/test_platform_compat.py::test_run_shell_timeout_kills_process_tree",
    ),
    (
        "M77 run_shell 风味又写死 Windows cmd",
        "agent/tools/builtin.py",
        """    if sys.platform == "win32":
        return "Windows cmd 语法，如 dir、type、python、pip"
    return "POSIX shell 语法（sh/bash），如 ls、cat、grep、python、pip\"""",
        """    if sys.platform == "win32":
        return "Windows cmd 语法，如 dir、type、python、pip"
    return "Windows cmd 语法，如 dir、type、python、pip"  # mutated: 又写死 cmd 方言""",
        "tests/test_platform_compat.py::test_shell_flavor_switches_with_platform",
    ),
    (
        "M78 run_shell 描述不接平台风味",
        "agent/tools/builtin.py",
        '                f"在本机执行 shell 命令（{_shell_flavor()}）。"',
        '                "在本机执行 shell 命令。"  # mutated: 描述丢掉平台风味',
        "tests/test_platform_compat.py::test_run_shell_description_carries_host_flavor",
    ),
    # ---------------- 第八轮审计（2026-10-29）：工具层密钥外流与 shell 包装绕过 ----------------
    (
        "M79 run_shell 输出不再过凭证掩码",
        "agent/tools/builtin.py",
        "            parts.append(mask_secrets(_decode(out_b))[-8000:])",
        "            parts.append(_decode(out_b)[-8000:])  # mutated: stdout 原样外发",
        "tests/test_audit8_security.py::test_run_shell_output_masks_secret_shaped_lines",
    ),
    (
        "M80 shell 不再拦密钥文件读取",
        "agent/tools/shell_rules.py",
        """        if _is_secret_basename(name):
            return f"命令拦截：`{name}` 是密钥文件，工具层不通过 shell 外发其内容，已拒绝\"""",
        """        if False:  # mutated: `type .env` 又能整份读出
            return f"命令拦截：`{name}` 是密钥文件，工具层不通过 shell 外发其内容，已拒绝\"""",
        "tests/test_audit8_security.py::test_secret_file_via_shell_blocked",
    ),
    (
        "M81 write_file 又允许覆写 .env",
        "agent/tools/builtin.py",
        """        if _is_secret_file(p):
            # 覆写 .env = 把 endpoint/key 改成外部地址，下一次加载会把全部凭据发往那里；""",
        """        if False:  # mutated: 写入侧不设守卫
            # 覆写 .env = 把 endpoint/key 改成外部地址，下一次加载会把全部凭据发往那里；""",
        "tests/test_audit8_security.py::test_write_and_append_tools_refuse_secret_files",
    ),
    (
        "M82 Runner 包装正文不再展开",
        "agent/tools/shell_safety.py",
        """            if _head_word(seg) not in _RUNNER_HEADS:
                continue""",
        """            if True:  # mutated: 只扫外层命令词，`"cmd" /c "del x"` 重新放行
                continue""",
        "tests/test_audit8_security.py::test_wrapped_dangerous_command_blocked",
    ),
    (
        "M83 重定向检查又要求操作符前有空白",
        "agent/tools/shell_rules.py",
        '    r"(?<![<>=!])"  # 排除 => / >= / -> 这类比较与箭头运算符粘在操作符前的写法',
        '    r"(?:^|\\s)"  # mutated: `echo a>..\\..\\evil` 少一个空格即写出项目根',
        "tests/test_audit8_security.py::test_redirect_without_space_still_checked",
    ),
    (
        "M84 异常文案不抹 URL userinfo",
        "agent/core/redact.py",
        '    (_URL_USERINFO, lambda m: m.group(1) + "****@" + m.group(2)),\n',
        "    # mutated: 无 query 段的 https://user:pass@host 整段外流\n",
        "tests/test_audit8_security.py::test_safe_error_text_masks_url_userinfo",
    ),
    # ---------------- 第八轮审计（2026-10-29）：状态一致性与契约（P1/P2） ----------------
    (
        "M85 下线写失败也落 disabled 标记",
        "agent/core/health.py",
        """            notice, retired_ok = self._notice_and_retire(name, model, models_env)
            if retired_ok:""",
        """            notice, retired_ok = self._notice_and_retire(name, model, models_env)
            if True:  # mutated: 先落标记再看不下线成没成 → 该模型永久不再重试下线""",
        "tests/test_audit8_fixes.py::test_disabled_flag_is_only_set_when_registry_write_landed",
    ),
    (
        "M86 落盘前不做写前重读比对（跨进程覆盖）",
        "agent/core/ledger.py",
        "                if raw is None or self._read_disk_text() == raw:",
        "                if True:  # mutated: 读盘与落盘之间被对端插队也照样整篇覆盖",
        "tests/test_audit8_fixes.py::test_write_remerges_when_peer_wrote_after_our_read",
    ),
    (
        "M87 档案落盘失败又静默吞掉",
        "agent/core/ledger.py",
        "                _warn_write_failed(self.store_path, e)",
        "                pass  # mutated: 一声不吭，档案从此不再更新没人知道",
        "tests/test_audit8_fixes.py::test_write_failure_is_reported_once_on_stderr",
    ),
    (
        "M88 健康账本键数不封顶（只增不减）",
        "agent/core/health.py",
        "        self._trim_models_locked()",
        "        pass  # mutated: 不裁剪，自动发现天天造的新键永久堆积",
        "tests/test_audit8_fixes.py::test_health_model_keys_are_capped",
    ),
    (
        "M89 并盘时不校验对端 fail_dates 元素类型",
        "agent/core/health.py",
        '                    | {d for d in (entry.get("fail_dates") or []) if isinstance(d, str)}',
        '                    | set(entry.get("fail_dates") or [])  # mutated: 整数日期混进来 → sorted() TypeError',
        "tests/test_audit8_fixes.py::test_non_string_fail_dates_are_dropped_on_load",
    ),
    (
        "M90 发现器不校验本站归属",
        "agent/core/node_discovery.py",
        '            if not _owns_station(adapter, info["base_url"]):',
        "            if False:  # mutated: 任何站都套用适配器写死的那份全局清单",
        "tests/test_audit8_fixes.py::test_discovery_skips_station_that_does_not_belong_to_adapter",
    ),
    (
        "M91 免费清单把付费条目判成下架",
        "agent/core/node_discovery.py",
        "        if m not in live_bare and (adapter is None or _manages(adapter, m))",
        "        if m not in live_bare  # mutated: 免费清单里没有的一律判下架",
        "tests/test_audit8_fixes.py::test_paid_entries_are_not_retired_by_a_free_only_list",
    ),
    (
        "M92 不可信远端 id 不清洗就进叠加层",
        "agent/core/node_discovery.py",
        "            live = _clean_models(live)",
        "            live = list(live)  # mutated: 含逗号的 id 进 env 等于注入多条模型",
        "tests/test_audit8_fixes.py::test_dirty_remote_ids_never_reach_the_overlay",
    ),
    (
        "M93 回退选路在判定与取值之间读两次池子",
        "agent/core/dispatch.py",
        """            if n not in tried and profiles.get(n) is not None and self._skip_reason(n) is None
        ]
        for name in self._latency_sorted(same):
            prof = profiles.get(name)""",
        """            if n not in tried and n in self.profiles and self._skip_reason(n) is None
        ]
        for name in self._latency_sorted(same):
            prof = self.profiles[name]  # mutated: 热同步换掉整本池子时 KeyError 穿出派工接口""",
        "tests/test_audit8_fixes.py::test_fallback_target_survives_profiles_being_swapped",
    ),
    (
        "M94 平票由完成先后而非候选编号决定",
        "agent/core/voting.py",
        "        best_idx, best_n = max(tally.items(), key=lambda kv: (kv[1], -kv[0]))",
        "        best_idx, best_n = max(tally.items(), key=lambda kv: kv[1])  # mutated: 平票取先出现者",
        "tests/test_audit8_fixes.py::test_ballot_tie_is_broken_by_candidate_number_not_completion_order",
    ),
    (
        "M95 select_best 的阈值只写进摘要不参与判定",
        "agent/core/voting.py",
        "        if best_n / total < th:\n            return None",
        "        if False:  # mutated: 未过半也当胜出方案返回\n            return None",
        "tests/test_audit8_fixes.py::test_select_best_honours_threshold_and_skips_empty_candidates",
    ),
    (
        "M96 select_best 空候选仍广播一轮投票",
        "agent/core/voting.py",
        "        if not candidates:\n            return None",
        "        if False:  # mutated: 空候选照样发一轮投票，白烧额度\n            return None",
        "tests/test_audit8_fixes.py::test_select_best_honours_threshold_and_skips_empty_candidates",
    ),
    (
        "M97 空正文以哨兵混进候选池",
        "agent/core/dispatch.py",
        '                "answer": str(resp.get("content") or ""),',
        '                "answer": resp.get("content", "") or "(空回复)",  # mutated: 真值过滤永不生效',
        "tests/test_audit8_fixes.py::test_empty_answer_stays_out_of_vote_candidates",
    ),
    (
        "M98 workers 显式传空被折叠成全池派工",
        "agent/bridge.py",
        """    if not names:
        return (
            None,
            "workers 显式传空：要么点名，要么省略该字段让池子自动挑（不能既写空又指望全池）",
        )""",
        """    if not names:
        return None, None  # mutated: 与"没点名"无法区分 → pick() 全池派工""",
        "tests/test_audit8_fixes.py::test_explicit_empty_workers_errors_instead_of_pool_wide_dispatch",
    ),
    (
        "M99 工具/技能出口不掩码就外发给外部主",
        "agent/bridge.py",
        '            text = mask_secrets(res["result"])',
        '            text = res["result"]  # mutated: 正文原样进 stdout 行协议',
        "tests/test_audit8_fixes.py::test_tool_and_skill_exits_are_masked_in_bridge",
    ),
    (
        "M100 MCP 外壳单条指令失败不带走整个会话",
        "agent/mcp_server.py",
        "        except Exception as e:  # noqa: BLE001 - 单条工具调用失败不能拖垮 MCP 会话",
        "        except KeyboardInterrupt as e:  # mutated: 常态异常不再兜底",
        "tests/test_audit8_fixes.py::test_mcp_shell_turns_a_handler_crash_into_ok_false",
    ),
    (
        "M101 call 单发模式崩了不回 JSON 行（构造失败也要回一行）",
        "agent/__main__.py",
        "    except Exception as e:  # noqa: BLE001 - 构造失败也要守住一行 JSON 的契约",
        "    except KeyboardInterrupt:  # mutated: 构造异常直接冒泡，stdout 只剩空行",
        "tests/test_audit9_fixes.py::test_call_mode_reports_init_failure",
    ),
    (
        "M102 探测先拿门再建客户端（构造抛错漏一格）",
        "agent/core/latency.py",
        """        try:
            client = self._probe_client_for(p)
        except Exception:  # noqa: BLE001 - 建不出探测客户端就跳过这个工人
            return None
        if not gate.acquire(timeout=_PROBE_GATE_WAIT):""",
        """        if not gate.acquire(timeout=_PROBE_GATE_WAIT):
            return None
        try:
            client = self._probe_client_for(p)
        except Exception:  # mutated: 门已拿到，抛在这里就永不 release""",
        "tests/test_audit8_fixes.py::test_probe_client_construction_failure_does_not_leak_station_slot",
    ),
    (
        "M103 stop_all 不等后台线程收尾（留孤儿子进程）",
        "agent/tools/mcp_client.py",
        "            thread.join(timeout=max(0.0, deadline - time.monotonic()))",
        "            pass  # mutated: 只叫醒不等收完，进程退出时子进程成孤儿",
        "tests/test_audit8_fixes.py::test_stop_all_waits_for_the_worker_threads_to_finish",
    ),
    (
        "M104 ${VAR} 解析为空不在加载时点名",
        "agent/config.py",
        '            _warn_empty_ref(station, "base_url", raw_station_item.get("base_url"), base_url)',
        "            pass  # mutated: 拖到运行期变成整站 401，再放大成清单自动下线",
        "tests/test_audit8_fixes.py::test_empty_reference_is_named_by_station_and_field",
    ),
    (
        "M105 discovery 又取插值前的占位符",
        "agent/config.py",
        '            discovery = str(item.get("discovery") or "").strip()',
        '            discovery = str(raw_station_item.get("discovery") or "").strip()  # mutated: ${VAR} 原样当适配器名',
        "tests/test_audit8_fixes.py::test_discovery_is_read_after_interpolation",
    ),
    (
        "M106 小写 .env 变量名不再补大写别名",
        "agent/config_load.py",
        "                if not upper_from_shell:\n"
        "                    os.environ[upper] = val\n"
        "                    _dotenv_keys.add(upper)",
        "                if False:  # mutated: 不补大写别名（Windows 巧合可用、Linux/CI 静默失效）\n"
        "                    pass",
        "tests/test_audit8_fixes.py::test_lowercase_dotenv_key_gets_upper_alias",
    ),
    # ---------------- 第九轮审计（2026-11-04）：安全层补漏 ----------------
    (
        "M107 PS 段前的危险首词不再检查（黑名单被分流短路）",
        "agent/tools/shell_safety.py",
        """    heads = _command_heads(stripped)
    if heads is None:
        return "命令拦截：Runner 包装嵌套过深，已拒绝（防绕过黑名单）\"""",
        """    heads = _command_heads(stripped)
    if heads is None:
        return "命令拦截：Runner 包装嵌套过深，已拒绝（防绕过黑名单）"
    heads = [h for h in heads if h in ("powershell", "pwsh")]  # mutated: 只看 PS 段""",
        "tests/test_audit9_fixes.py::test_ps_branch_no_longer_short_circuits_head_blacklist",
    ),
    (
        "M108 `>|` 重定向目标不做越界检查",
        "agent/tools/shell_rules.py",
        '    r"(?:>>|>\\||<>|>|<)"  # 操作符：长操作符优先匹配（>> 先于 >、>| 先于 >）',
        '    r"(?:>>|>|<)"  # mutated: `>|` 整条 match 失败、越界检查落空',
        "tests/test_audit9_fixes.py::test_clobber_redirect_targets_are_checked",
    ),
    (
        "M109 内联代码拼接读密钥文件不拦",
        "agent/tools/shell_safety.py",
        "    for pat, why in _CODE_SECRET_READ:",
        "    for pat, why in []:  # mutated: 拼接读 .env 的形态放行",
        "tests/test_audit9_fixes.py::test_inline_code_cannot_concat_read_secret_file",
    ),
    (
        "M110 密钥名子串判定撤掉（KEY_ID 类命名漏出）",
        "agent/core/redact.py",
        "    return any(word in up for word in SECRET_NAME_SUBSTRINGS)",
        "    return False  # mutated: 只认末段词，AWS_ACCESS_KEY_ID 原样外流",
        "tests/test_audit9_fixes.py::test_mask_covers_midword_secret_names_and_dict_repr",
    ),
    (
        "M111 Runner 展开只剥一层（超深包装正文不再扫描）",
        "agent/tools/shell_safety.py",
        "            queue.append(payload)",
        "            continue  # mutated: 不再深入展开，第 2 层起包装正文永远进不了扫描",
        "tests/test_audit9_fixes.py::test_runner_nesting_beyond_old_depth_cap_is_blocked",
    ),
    # ---------------- 第九轮审计（2026-11-04）：配置层 ----------------
    (
        "M112 配置不完整的站照常派工（broken 标记不参与跳过判定）",
        "agent/core/orchestrator.py",
        """        p = self.profiles.get(name)
        if p is not None and getattr(p, "broken", ""):""",
        """        p = self.profiles.get(name)
        if p is not None and False:  # mutated: 空端点站照常派工，必败请求污染健康档案""",
        "tests/test_audit9_fixes.py::test_broken_worker_is_not_dispatched_and_health_says_so",
    ),
    (
        "M113 非有限数值配置崩启动（int(inf) 的 OverflowError 不接）",
        "agent/config_util.py",
        "    except (TypeError, ValueError, OverflowError):",
        "    except (TypeError, ValueError):  # mutated: inf 走 OverflowError 直接炸",
        "tests/test_audit9_fixes.py::test_numeric_config_rejects_nonfinite",
    ),
    # ---------------- 第九轮审计（2026-11-04）：协作内核 ----------------
    (
        "M114 重试链不整体记在飞（退避间隙被回收即关连接池）",
        "agent/core/llm.py",
        """        self._enter_call()
        try:
            yield self""",
        """        try:
            yield self  # mutated: 重试链不记在飞，close_when_idle 当场关""",
        "tests/test_audit9_fixes.py::test_call_span_defers_close_until_chain_ends",
    ),
    (
        "M115 健康档案并盘不验 disabled 类型（混型 TypeError 吞隔离）",
        "agent/core/health.py",
        """                days = [
                    d
                    for d in (entry.get("disabled"), mine.get("disabled"))
                    if isinstance(d, str) and d
                ]""",
        """                days = [d for d in (entry.get("disabled"), mine.get("disabled")) if d]  # mutated: min() 混型抛 TypeError""",
        "tests/test_audit9_fixes.py::test_health_merge_tolerates_nonstring_disabled",
    ),
    (
        "M116 叠加层写前不重读（并发进程整篇互相覆盖）",
        "agent/core/node_discovery.py",
        "            disk = load_overlay(self.overlay_path)",
        "            disk = {}  # mutated: 不重读盘，其它站的增量被整篇抹掉",
        "tests/test_audit9_fixes.py::test_run_once_merges_into_overlay_written_by_another_process",
    ),
    (
        "M117 探测一人一线程（池大时线程爆炸）",
        "agent/core/latency.py",
        "        slots = max(1, min(self._concurrency, len(names_unique)))",
        "        slots = len(names_unique)  # mutated: 一人一线程",
        "tests/test_audit9_fixes.py::test_probe_round_uses_bounded_threads",
    ),
    (
        "M118 整体限时不算回退链（回退中的工人被误判超时）",
        "agent/core/parallelism.py",
        '        chain = 1 + max(0, int(getattr(self, "_fallback_models", 0) or 0))',
        "        chain = 1  # mutated: 只按单模型重试链折算",
        "tests/test_audit_fixes.py::test_effective_timeout_scales_with_batches_and_respects_explicit",
    ),
    (
        "M119 延迟裁剪浮点截断差一位（ratio=0.8/n=5 完全不裁）",
        "agent/core/latency.py",
        "        drop = math.floor(len(ordered) * (1.0 - self._latency_keep_ratio) + 1e-9)",
        "        drop = int(len(ordered) * (1.0 - self._latency_keep_ratio))  # mutated: 截断差一位",
        "tests/test_audit9_fixes.py::test_keep_ratio_integer_boundary_not_lost",
    ),
]


def _child_env() -> dict:
    """子进程统一按 UTF-8 收发：Windows 上 `text=True` 默认按本地代码页（cp936）解码，
    而 pytest 的输出里带着中文用例名/断言文本——2026-10-29 那次就跑出了
    `UnicodeDecodeError: 'gbk' codec can't decode byte 0x80`（崩在 subprocess 的 reader
    线程里）。指定 encoding 之外再把 PYTHONIOENCODING 钉住，两侧行为一致。
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_pytest(node: str) -> int:
    """跑一个用例节点，返回 pytest 退出码。

    0 全过；1 有用例失败（真断言失败）；2 中断/收集错；3 内部错；4 用法错；5 未收集到用例。
    返回值必须是退出码本身（而不是布尔）：调用方要靠 0 / 1 / 其它 三分类判定"变异是否
    真的被用例挡住"——旧写法只判 `!= 0`，把"收集不到用例""语法炸掉"也一并算成有效守卫。
    """
    proc = subprocess.run(
        [PY, "-m", "pytest", "-q", "-x", node],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_child_env(),
        timeout=600,
    )
    return proc.returncode


def node_collects(node: str) -> bool:
    """该用例节点是否真的存在、能收到用例。

    为什么必须先查（审计 P1，假安全的根源）：`run_pytest` 只以 `returncode != 0` 认定
    "变异后失败 = 用例有效"，而 pytest 在节点不存在/被改名时返回 4、收集到 0 条时返回 5，
    两者都 !=0 → 变异条目会被记成 OK。也就是说**用例一旦消失，这条变异门禁会立刻变成
    永不再失败的绿项**，保护层悄悄没了却没人报警。先 collect-only 确认节点还在。
    """
    proc = subprocess.run(
        [PY, "-m", "pytest", "-q", "--collect-only", node],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_child_env(),
        timeout=300,
    )
    return proc.returncode == 0


# 上一轮被硬杀时可能把源码留在变异态（本项目真发生过，`.tmp_prestore/` 就是那种残留）：
# 开跑前先扫全部 agent/ 与 tests/ 源文件，带着 `# mutated:` 标记就拒绝在坏态上继续改。
# 旧版只扫 MUTATIONS 的目标文件，别的文件被留在坏态不会被发现；本脚本自身要排除——
# MUTATIONS 里本来就写着 `# mutated:` 字样，不排除会误判。
_stale_scope = [
    p
    for p in list((ROOT / "agent").rglob("*.py")) + list((ROOT / "tests").rglob("*.py"))
    if p.resolve() != Path(__file__).resolve()
]
stale = sorted(
    str(p.relative_to(ROOT)) for p in _stale_scope if "# mutated:" in p.read_text(encoding="utf-8")
)
if stale:
    print(f"[中止] 这些文件仍带着上次变异留下的 `# mutated:` 标记，先从 git 还原再跑：{stale}")
    raise SystemExit(1)

rows: list[tuple[str, str, str]] = []
for label, rel, old, new, node in MUTATIONS:
    path = ROOT / rel
    original = path.read_text(encoding="utf-8")
    if old not in original:
        rows.append((label, "SKIP", "锚点未找到（源码已变）"))
        continue
    if not node_collects(node):
        rows.append((label, "FAIL", "用例节点不存在（被改名/删除，这条变异已失去守卫）"))
        continue
    mutated = original.replace(old, new, 1)
    try:
        # 变异体自己就得是合法 Python：否则"用例失败"只是语法/导入错，不能证明断言有效
        # （审计：旧写法把这种失败也记 OK，等于拿坏源码冒充回归保护）。
        ast.parse(mutated)
    except SyntaxError as exc:
        rows.append((label, "BROKEN", f"变异体语法错误（{exc.msg}）：不能据此判定用例有效性"))
        continue
    try:
        path.write_text(mutated, encoding="utf-8")
        code = run_pytest(node)
    finally:
        path.write_text(original, encoding="utf-8")
    if code == 0:
        rows.append((label, "TAUTOLOGY", "变异后仍通过=恒真用例"))
    elif code == 1:
        rows.append((label, "OK", "变异后失败=用例有效"))
    else:
        rows.append((label, "BROKEN", f"变异后 pytest 退出码 {code}（未真正跑用例，不能判定）"))

print("=" * 78)
print(f"{'变异':<26}{'结果':<12}说明")
print("-" * 78)
for label, verdict, note in rows:
    print(f"{label:<26}{verdict:<12}{note}")
bad = [r for r in rows if r[1] != "OK"]
print("=" * 78)
print("全部用例均为有效断言（去掉修复即失败）" if not bad else f"存在可疑用例：{bad}")
sys.exit(1 if bad else 0)
