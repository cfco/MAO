"""全局测试夹具：为每个用例隔离模型健康档案，避免共享落盘文件造成跨用例污染。

背景：ModelHealth 默认写到仓库真实路径 data/model_health.json。多个测试
（test_core 的失败重试、test_health、test_smoke……）都会触发失败记账，
"当日隔离"会让后续用例里同名工人（如 dummy、node-a:agnes-2.5-flash）被
直接跳过、FakeClient 永远不被调用，表现为断言失败或线程挂死。
每个用例独享 tmp 档案后，隔离状态互不干扰，也不会弄脏用户真实档案。
"""

import pytest


@pytest.fixture(autouse=True)
def isolate_model_health_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MAO_HEALTH_FILE", str(tmp_path / "model_health.json"))
    # 自动下线改写目标（model_registry.txt）同样隔离：worker 集成测试连跑 7 个
    # 运行日失败也可能触发 retire，绝不能落到真实仓库的模型清单上。
    monkeypatch.setenv("MAO_MODEL_REGISTRY_FILE", str(tmp_path / "model_registry.txt"))
    # 延迟档案同理隔离：延迟样本会影响 pick() 的候选裁剪（问题5），跨用例串样本会让
    # "轮转顺序/全员参与"这类断言随机失败。
    monkeypatch.setenv("MAO_LATENCY_FILE", str(tmp_path / "model_latency.json"))
    # 并整轮关掉延迟自动探测：默认开启时首个派工就会起后台守护线程真打网络，
    # 离线用例必须完全无网络。测延迟探测本身的用例直接构造 LatencyProber（不经配置）。
    monkeypatch.setenv("MAO_LATENCY_PROBE", "0")
    # 免费节点自动发现同样必须隔离（审计 P0）：config.yaml 的 node-b 写了 `discovery: kilo`，
    # 而 collaboration.free_node_discovery 默认 true，WorkerPool 建池即起发现线程。少了这两行，
    # "全离线"承诺会悄悄失效——实测发生过：一次 `uv run pytest` 之后真打了 api.kilo.ai，
    # 并把发现结果写进仓库内的 data/free_nodes.json，而该文件又经 apply_overlay 反过来
    # 改变下一次真实运行的池子（新增/移出模型）。
    monkeypatch.setenv("MAO_FREE_NODE_PROBE", "0")
    monkeypatch.setenv("MAO_FREE_NODES_FILE", str(tmp_path / "free_nodes.json"))


@pytest.fixture(autouse=True)
def zero_dispatch_backoff(monkeypatch):
    """把派工重试的退避时长归零（只影响"睡多久"，不动"退避几次、按什么顺序"）。

    第五轮审计把重试环从 `LLMClient.chat()` 挪到了编排层的 `_attempt_with_retry`，
    退避不再由测试里 patch 的 `client._backoff` 管：整套从 17s 涨到 214s（7 个用例
    各真睡 2/4/8 秒）。CI 的 ubuntu 同样会白付这三分钟。要验退避**顺序/是否发生**的
    用例自己 patch `time.sleep` 记事件（见 test_audit5 的门序用例），别依赖真实秒数。
    """
    import agent.core.orchestrator as orch

    monkeypatch.setattr(orch, "backoff_seconds", lambda attempt, error_type: 0.0)
