"""第九轮审计（2026-11-04）修复的回归用例。

覆盖四块安全修复：
1. powershell 分流不再短路首词黑名单（`del … & powershell -c` 必须拦）；
2. `>|` / `<>` 重定向目标必须过越界检查；
3. 内联代码拼接读密钥文件（`open('.en'+'v')`）/ 读后编码外带必须拦；
4. 密钥名词表子串化 + 掩码口径（KEY_ID / dict repr / 任意 scheme userinfo）。

全部离线：只用纯函数判定与环境变量，不发网络、不建子进程。
"""

from __future__ import annotations

import pytest

from agent.core.redact import mask_secrets
from agent.tools import builtin
from agent.tools.shell_safety import _check_command_safety

# ---------------- 1. PS 分流不再短路黑名单（P0） ----------------


def test_ps_branch_no_longer_short_circuits_head_blacklist():
    """危险段排在 powershell 段之前也必须拦：黑名单对全部 heads 无条件生效。

    旧实现 heads 里出现 powershell 就只扫 -Command 正文，`del /f /q x &
    powershell -c "Get-Date"` 的 del 既不在 body 里、也不再被检查（实测放行）。
    """
    assert _check_command_safety('del /f /q x & powershell -c "Get-Date"') is not None
    assert _check_command_safety('format D: /q & pwsh -c "1"') is not None
    assert _check_command_safety('shutdown /r /t 0 & powershell -nop -c "1"') is not None
    # POSIX 词同样生效
    assert _check_command_safety('rm -rf / & pwsh -c "1"') is not None


def test_ps_legit_commands_still_pass():
    """正常 PowerShell 用法不受影响；危险内容藏在 -Command 正文里照样被扫出来。"""
    assert _check_command_safety('powershell -Command "Get-Date"') is None
    assert _check_command_safety("powershell -Command \"Write-Output 'hello world'\"") is None
    # PS 危险模式仍由正文扫描兜住
    assert _check_command_safety('powershell -Command "Remove-Item -Force x"') is not None
    assert _check_command_safety('powershell -Command "iex($(content))"') is not None


# ---------------- 2. `>|` / `<>` 重定向越界（P1） ----------------


def test_clobber_redirect_targets_are_checked():
    """`>|`（POSIX clobber）与 `<>` 的目标必须过越界检查。

    旧操作符表只有 `>> > <`，而目标字符类排除 `|`，`echo x >| /etc/evil`
    整条 match 失败、绝对路径直写盘外（实测放行）。
    """
    assert _check_command_safety("echo x >| /etc/evil") is not None
    assert _check_command_safety("echo x >| ..\\..\\evil.txt") is not None
    assert _check_command_safety("echo x <> /etc/evil") is not None
    # 正常重定向不受影响
    assert _check_command_safety("echo x > out.txt") is None
    assert _check_command_safety("echo x >> out.txt") is None


# ---------------- 3. 内联代码拼接读密钥文件（P1） ----------------


def test_inline_code_cannot_concat_read_secret_file():
    """`open('.en'+'v')` 拼接绕过文件名识别的写法必须拦。"""
    assert (
        _check_command_safety(
            "python -c \"import base64;print(base64.b64encode(open('.en'+'v','rb').read()).decode())\""
        )
        is not None
    )
    assert _check_command_safety("python -c \"print(open('.'+'env').read())\"") is not None
    # node 同样拦
    assert (
        _check_command_safety(
            "node -e \"console.log(require('fs').readFileSync('.'+'env','utf8'))\""
        )
        is not None
    )
    # 字面量 .env 本来就拦（既有规则，防止改崩）
    assert _check_command_safety("python -c \"print(open('.env').read())\"") is not None


def test_inline_code_open_and_encode_is_blocked():
    """读文件 + base64/codecs 编码的组合拦下：读什么不重要，编码外带意图明确。"""
    assert (
        _check_command_safety(
            "python -c \"import base64;print(base64.b64encode(open('cfg.txt','rb').read()))\""
        )
        is not None
    )


def test_inline_code_legit_file_reads_still_pass():
    """正常开发内联代码不误伤：不带拼接、不碰编码组合的 open 照常放行。"""
    assert _check_command_safety("python -c \"print(open('requirements.txt').read())\"") is None
    assert _check_command_safety("python -c \"print(open('data.txt','rb').read())\"") is None
    assert (
        _check_command_safety("python -c \"import json;print(json.load(open('c.json')))\"") is None
    )


# ---------------- 4. 密钥名子串判定 + 掩码补漏（P1） ----------------


def test_mask_covers_midword_secret_names_and_dict_repr():
    """`..._KEY_ID` / `..._TOKEN_V2` / CREDENTIALS / dict repr 形态都必须掩码。"""
    assert "abcdefgh123456" not in mask_secrets("AUTH_TOKEN_V2=abcdefgh123456")
    assert "AUTH_TOKEN_V2=****" in mask_secrets("AUTH_TOKEN_V2=abcdefgh123456")
    assert "plainsecretvalue" not in mask_secrets("CREDENTIALS=plainsecretvalue")
    # Python dict / JSON repr：名字与分隔符之间隔着引号也要命中
    out = mask_secrets("cfg = {'NODE_A_KEY': 'plainvalue123'}")
    assert "plainvalue123" not in out, out
    assert "****" in out, out


def test_mask_covers_non_http_url_userinfo():
    """postgres://u:p@h/ 这类连接串里的密码也要掩码（scheme 不再限定 http/ws）。"""
    out = mask_secrets("DATABASE_URL=postgres://user:secretpw@db.host:5432/app")
    assert "secretpw" not in out, out
    assert "db.host" in out, out  # host 保留，排障还要看
    # 正常 URL 不误伤
    keep = mask_secrets("see https://example.com/@user for info")
    assert "example.com" in keep, keep


def test_child_env_strips_midword_secret_names(monkeypatch):
    """`AWS_ACCESS_KEY_ID` / `AUTH_TOKEN_V2` / `CREDENTIALS` 不再漏进子进程环境。"""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "v1")
    monkeypatch.setenv("AUTH_TOKEN_V2", "v2")
    monkeypatch.setenv("CREDENTIALS", "v3")
    monkeypatch.setenv("NODE_A_KEY", "v4")
    monkeypatch.setenv("PATH", "kept")
    env = builtin._child_env()
    assert "AWS_ACCESS_KEY_ID" not in env
    assert "AUTH_TOKEN_V2" not in env
    assert "CREDENTIALS" not in env
    assert "NODE_A_KEY" not in env
    assert env.get("PATH") == "kept"


# ---------------- 5. Runner 展开预算：超深包装宁可整条拒绝（P2） ----------------


def test_runner_nesting_beyond_old_depth_cap_is_blocked():
    """`cmd /c` 套 5 层以上也要拦：旧实现深度限 4，第 5 层起黑名单失效（实测放行）。"""
    deep5 = "cmd /c " * 5 + "del /f /q x"
    deep8 = "cmd /c " * 8 + "del /f /q x"
    assert _check_command_safety(deep5) is not None
    assert _check_command_safety(deep8) is not None
    assert _check_command_safety('cmd /c "del /f /q x"') is not None


def test_runner_normal_nesting_still_passes():
    """正常一两层包装不受影响。"""
    assert _check_command_safety("cmd /c echo hello") is None
    assert _check_command_safety('sh -c "echo hi"') is None


# ---------------- 6. 配置不完整的站打 broken 标记（P0） ----------------


def test_config_marks_stations_with_missing_base_url_or_empty_ref():
    """空端点 / 引用变量解析成空 key 的站打 broken 标记；字面空 key（本地 Ollama）不受影响。

    站照常入池（分层契约与诊断视图依赖 profile 存在），但派工/回退/health 全按
    不可用处理：旧实现只警告不拦，实测 19 个工人里 16 个端点/密钥为空串照常入池，
    health 还把它们全报 available:true——派工必败且污染健康/延迟档案。
    """
    from agent.config import Config, _interpolate

    raw = {
        "agents": [
            {"name": "good", "base_url": "http://x/v1", "api_key": "k", "models": "m1@8k"},
            {"name": "nourl", "base_url": "", "api_key": "k", "models": "m2@8k"},
            {
                "name": "nokeyref",
                "base_url": "http://y/v1",
                "api_key": "${MAO_AUDIT9_MISSING_KEY}",
                "models": "m3@8k",
            },
            # 字面空串 = 本地免 key 站的合法写法，必须不受影响
            {"name": "emptykeylit", "base_url": "http://z/v1", "api_key": "", "models": "m4@8k"},
        ]
    }
    cfg = Config(_interpolate(raw), raw=raw)
    broken = {p.name: p.broken for p in cfg.agent_profiles}
    assert broken["good"] == "" and broken["emptykeylit"] == ""
    assert "base_url" in broken["nourl"]
    assert "api_key" in broken["nokeyref"]


def test_broken_worker_is_not_dispatched_and_health_says_so():
    """broken 工人：点名派工被拦、health 报 available:false 且带原因。"""
    from agent.config import Config, _interpolate
    from agent.core.orchestrator import WorkerPool

    raw = {
        "agents": [
            {"name": "nourl", "base_url": "", "api_key": "k", "models": "m1@8k"},
            {"name": "ok", "base_url": "http://x/v1", "api_key": "k", "models": "m1@8k"},
        ]
    }
    pool = WorkerPool(Config(_interpolate(raw), raw=raw))
    out = pool.ask_result("nourl", "hi")
    assert out["ok"] is False and "配置不完整" in out["error"], out
    status = {w["worker"]: w for w in pool.health_status()}
    assert status["nourl"]["available"] is False
    assert "base_url" in status["nourl"]["broken"]
    assert status["ok"]["available"] is True and status["ok"]["broken"] == ""
    # 可派工人不进回退候选：_skip_reason 统一承接
    assert pool._skip_reason("nourl") is not None
    assert pool._skip_reason("ok") is None


# ---------------- 7. 数值配置拒绝非有限值（P1） ----------------


def test_numeric_config_rejects_nonfinite():
    """`cooldown_max: .inf` 这类写法不能崩启动：inf/nan 一律警告 + 回退默认。

    旧实现 int(float('inf')) 抛 OverflowError 直接炸 WorkerPool 构造。
    """
    from agent.config import Config

    assert Config.as_int(float("inf"), "k.inf", 7, minimum=1) == 7
    assert Config.as_int(float("nan"), "k.nan", 7) == 7
    assert Config.as_float(float("inf"), "k.fin", 5.0) == 5.0
    assert Config.as_float(float("nan"), "k.fnan", 5.0) == 5.0
    # 正常值不受影响
    assert Config.as_int("3", "k.ok", 7, minimum=1) == 3
    assert Config.as_float("1.5", "k.fok", 5.0) == 1.5


# ---------------- 8. 重试链整体记在飞：退避间隙不许关 client（P1） ----------------


def test_call_span_defers_close_until_chain_ends():
    """重试链中途被热同步点名回收，client 不能当场关：退避后的重试还要用它。"""
    from agent.core.llm import LLMClient

    c = LLMClient("http://127.0.0.1:9/v1", "k", "m")
    with c.call_span():
        c.close_when_idle()  # 链中间被 sync_free_nodes 回收
        assert c._closed is False, "重试链还没收尾就 close，后续尝试会打在已关闭的连接池上"
    assert c._closed is True, "链收尾后应统一关闭（close_pending 生效）"


# ---------------- 9. 健康档案并盘/读盘容错（P1） ----------------


def test_health_merge_tolerates_nonstring_disabled(tmp_path):
    """对端档案 disabled 写成整数：min() 混型曾抛 TypeError 并被整段吞掉，
    当日隔离静默不落盘。并盘只收字符串日期。"""

    from agent.core.health import ModelHealth

    h = ModelHealth(tmp_path / "mh.json")
    with h._lock:
        h._data = {"m1": {"fail_dates": [], "disabled": "2026-11-01"}}
        h._merge_disk_locked(
            {
                "models": {
                    "m1": {"fail_dates": [], "disabled": 123},
                    "m2": {"fail_dates": [2026], "disabled": "2026-11-02"},
                }
            }
        )
        assert h._data["m1"]["disabled"] == "2026-11-01"
        assert h._data["m2"]["disabled"] == "2026-11-02"
        assert h._data["m2"]["fail_dates"] == []


def test_health_load_drops_nonstring_dates(tmp_path):
    """读盘同样按元素验类型：整数 fail_dates/disabled 被丢弃而不是炸记账。"""
    import json

    from agent.core.health import ModelHealth

    p = tmp_path / "mh.json"
    p.write_text(
        json.dumps(
            {
                "version": 2,
                "active_dates": [1],
                "models": {"m": {"fail_dates": [7], "disabled": 5}},
            }
        ),
        encoding="utf-8",
    )
    h = ModelHealth(p)
    h._ensure_loaded()
    assert h._data["m"]["fail_dates"] == []
    assert "disabled" not in h._data["m"]


# ---------------- 10. 叠加层写前重读合并 + 集合比较（P1/P2） ----------------


def test_merge_deltas_unions_both_sides():
    """增量合并：added 保序取并集、retired 排序取并集（跨进程稳定）。"""
    from agent.core.node_discovery import _merge_deltas

    merged = _merge_deltas(
        {"added": ["a@1"], "retired": ["r1"]},
        {"added": ["b@2"], "retired": ["r1"]},
    )
    assert merged["added"] == ["a@1", "b@2"]
    assert merged["retired"] == ["r1"]


def test_delta_signature_ignores_order():
    """retired 来自 set 迭代、added 跟远端顺序：顺序抖动不算变化，不该重写叠加层。"""
    from agent.core.node_discovery import _delta_signature

    a = {"added": ["x@1", "y@2"], "retired": ["r1", "r2"]}
    b = {"added": ["y@2", "x@1"], "retired": ["r2", "r1"]}
    assert _delta_signature(a) == _delta_signature(b)


def test_run_once_merges_into_overlay_written_by_another_process(tmp_path):
    """另一进程刚写入的其它站增量必须幸存：run_once 写前重读、按站合并。"""
    import json
    import types as _t

    from agent.core.node_discovery import Adapter, FreeModel, FreeNodeDiscoverer

    reg = tmp_path / "registry.txt"
    reg.write_text("NODE_X_MODELS=m1@8k\nNODE_Y_MODELS=y1@8k\n", encoding="utf-8")
    overlay = tmp_path / "free_nodes.json"
    overlay.write_text(
        json.dumps({"NODE_Y_MODELS": {"added": ["ynew@8192"], "retired": []}}),
        encoding="utf-8",
    )

    class _A(Adapter):
        name = "x"
        host = ""
        free_marker = ""

        def fetch(self, base_url, api_key, timeout):
            return [FreeModel("m1", 8192), FreeModel("mnew", 8192)]

    cfg = _t.SimpleNamespace(
        free_node_interval=86400.0,
        free_node_timeout=10.0,
        free_node_discovery=True,
        model_registry_path=reg,
        agent_profiles=[
            _t.SimpleNamespace(
                models_env="NODE_X_MODELS",
                base_url="http://x/v1",
                api_key="",
                discovery="x",
            )
        ],
    )
    d = FreeNodeDiscoverer(cfg, overlay, adapters={"x": _A()})
    d.run_once()
    data = json.loads(overlay.read_text(encoding="utf-8"))
    assert data["NODE_Y_MODELS"]["added"] == ["ynew@8192"], (
        "其它站的增量被整篇抹掉（跨进程 lost update）"
    )
    assert "mnew@8192" in data["NODE_X_MODELS"]["added"]


# ---------------- 11. 探测轮固定线程槽（P2） ----------------


def test_probe_round_uses_bounded_threads(monkeypatch, tmp_path):
    """探测轮线程数 = 并发上限：旧实现一人一线程，池几十个模型时线程爆炸。"""
    from agent.core import latency as lat_mod

    store = lat_mod.LatencyStore(tmp_path / "lat.json")
    prober = lat_mod.LatencyProber(store, 600.0, lambda nm: 1.0)
    created: list[int] = []
    real_thread = lat_mod.threading.Thread

    def counting_thread(*a, **k):
        created.append(1)
        return real_thread(*a, **k)

    monkeypatch.setattr(lat_mod.threading, "Thread", counting_thread)
    prober._run_round([f"w{i}" for i in range(20)])
    assert len(created) <= prober._concurrency, (
        f"探测轮起了 {len(created)} 条线程，超过并发上限 {prober._concurrency}"
    )


# ---------------- 12. 延迟裁剪整数边界（P2） ----------------


def test_keep_ratio_integer_boundary_not_lost():
    """ratio=0.8、n=5 应裁掉最慢 1 个：浮点截断曾把 0.999… 截成 0（完全不裁剪）。"""
    from agent.core.latency import LatencyMixin

    lat = {"w1": 100.0, "w2": 200.0, "w3": 300.0, "w4": 400.0, "w5": 500.0}

    class _latency:
        @staticmethod
        def snapshot():
            return dict(lat)

    class _Stub:
        _latency_keep_ratio = 0.8
        _latency_min_samples = 2
        latency = _latency
        _latency_sorted = LatencyMixin._latency_sorted
        _latency_window = LatencyMixin._latency_window

    kept = _Stub()._latency_window(["w1", "w2", "w3", "w4", "w5"])
    assert len(kept) == 4, f"ratio=0.8、n=5 应裁掉最慢 1 个：{kept}"
    assert "w5" not in kept


# ---------------- 13. call 外壳构造失败也回一行 JSON（P3） ----------------


def test_call_mode_reports_init_failure(monkeypatch, capsys):
    """`mao call` 的契约是"一次调用一行回包"：构造（load_config/Bridge）抛异常
    也不能只剩 stderr 崩栈，要回一行 ok:false（第九轮前构造在 try 之外）。"""
    import json as _json

    import agent.__main__ as m

    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(m, "load_config", _boom)
    # payload 必须走 argv（省略时会去读 stdin，pytest 捕获下必炸）
    monkeypatch.setattr("sys.argv", ["agent", "call", "ping", "{}"])
    with pytest.raises(SystemExit) as ei:
        m.main()
    assert ei.value.code == 1
    resp = _json.loads(capsys.readouterr().out.strip())
    assert resp["ok"] is False and "初始化失败" in resp["error"]
