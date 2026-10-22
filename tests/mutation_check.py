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
        "        return batches * worst + 30.0",
        "        return 300.0",
        f"{TESTFILE}::test_effective_timeout_scales_with_batches_and_respects_explicit",
    ),
    (
        "M5 MCP 连接不再共享",
        "agent/tools/mcp_client.py",
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
        """                text = self._snapshot_locked()
            try:
                atomic_write_text(self.store_path, text)
            except OSError:
                pass  # 档案写不进去只损失持久化，内存账本照常工作""",
        """                text = self._snapshot_locked()
                try:  # mutated: 落盘 IO 回到记账锁内
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
        "agent/core/dispatch.py",
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
            disk = self._disk_for_merge()  # IO 留在 _lock 之外
            self._ensure_loaded()  # 载入同理锁外：合并/取快照前账本必已就绪（子类实现）
            with self._lock:
                if disk is not None:
                    self._merge_disk_locked(disk)
                text = self._snapshot_locked()""",
        """        disk = self._disk_for_merge()  # mutated: 快照在写锁外取，写盘顺序可能反于快照顺序
        self._ensure_loaded()  # 载入同理锁外：合并/取快照前账本必已就绪（子类实现）
        with self._lock:
            if disk is not None:
                self._merge_disk_locked(disk)
            text = self._snapshot_locked()
        with self._write_lock:""",
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
        "            disk = self._disk_for_merge()  # IO 留在 _lock 之外\n"
        "            self._ensure_loaded()  # 载入同理锁外：合并/取快照前账本必已就绪（子类实现）\n"
        "            with self._lock:\n"
        "                if disk is not None:\n"
        "                    self._merge_disk_locked(disk)\n",
        "            self._ensure_loaded()  # 载入同理锁外：合并/取快照前账本必已就绪（子类实现）\n"
        "            with self._lock:\n",
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
        "        if _is_secret_file(p):",
        "        if False:  # mutated: 允许根之内即可读出 .env",
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
        "agent/tools/mcp_client.py",
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
        "agent/tools/shell_safety.py",
        r'    low = parts[0].replace("^", "").replace("\\", "/").rsplit("/", 1)[-1].lower()'
        "  # 去路径前缀",
        '    low = parts[0].replace("\\\\", "/").rsplit("/", 1)[-1].lower()  # mutated: de^l 放行',
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
        "agent/tools/shell_safety.py",
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
        "        worst = worst_case_per_request(per_call, max_retries)",
        "        worst = per_call  # mutated: 只按单次 timeout 折算，尾部批次被误判未完成",
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
        "agent/tools/builtin.py",
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
        "agent/tools/builtin.py",
        """        if name.rsplit("_", 1)[-1] in SECRET_NAME_TAILS or any(
            name.endswith(tail) for tail in SECRET_NAME_TAILS
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
        """        for h in heads:
            if _is_dangerous_word(h):
                return f"命令拦截：检测到危险命令 `{h}`\"""",
        """        for h in heads:
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
]


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
