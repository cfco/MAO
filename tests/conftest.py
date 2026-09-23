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
