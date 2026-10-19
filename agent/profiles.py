"""智能体池的成员定义（AgentProfile）。

从 config.py 拆出（2026-10-19，为守「单文件 ≤500 行」约定）：本类只描述"池里一条工人"
有哪些字段、站键怎么算、对外怎么展示，不含加载流程与 Config 装配，因此可以独立成模块，
被 dispatch / batch / free_nodes / orchestrator 直接引用而不必拖上整个配置加载器。
"""

from __future__ import annotations

from dataclasses import dataclass

# 未标注上下文长度的模型在对外展示（list_agents / brief）时按此值兜底。
# 注意：历史裁剪已随内部主循环移除，本值**不参与**任何 prompt 裁剪/选路决策。
DEFAULT_CONTEXT = 128_000


@dataclass
class AgentProfile:
    """智能体池中的一条：一个可 API 接入的模型智能体。

    支持「一站多模型」：一个站（base_url+api_key）可挂多个 model，
    每个模型展开成一个 AgentProfile。名字：单模型取站名，多模型取"站名:模型名"。
    """

    name: str
    base_url: str
    api_key: str
    model: str
    context_length: int = 0  # 上下文窗口(token)；0=未指定按 DEFAULT_CONTEXT
    note: str = ""
    # 能力标签（逗号分隔，如 "code,中文,长上下文"）：外部主据此决定把哪类活派给谁。
    tags: str = ""
    # 该模型清单来自 model_registry.txt 里的哪个变量（如 NODE_A_MODELS），
    # 便于诊断"这批模型挂在哪一站"，也是自动下线改写（加 #）的目标行定位用。
    # 值写死在 config.yaml（非 ${VAR}）时为空。
    models_env: str = ""
    # 免费节点发现适配器名（"kilo"/"bynara"/""=不自动发现）。由 config.yaml 各 agent
    # 的 `discovery:` 字面量指定；发现器据此拉该站在线免费清单并与 registry 对账。
    discovery: str = ""

    @property
    def station_key(self) -> tuple[str, str]:
        """站键：同站 = 同 (base_url, api_key)。**全项目唯一口径**。

        延迟档案、站门限流、同站分组/回退都拿它当 key。用 (base_url, api_key) 而非仅
        base_url：不同站可能共用同一 base_url（镜像 / 多租户），api_key 才真正区分站。
        集中一处后，将来若改站身份口径（如 base_url 归一化）只改这里，不会有的地方
        认旧键、有的地方认新键而对不上（原先 9 处各拼一次 (p.base_url, p.api_key)）。
        """
        return (self.base_url, self.api_key)

    def brief(self) -> dict:
        """对外展示用（不含 key）。context_length 仅用于展示/预检，不参与选路。"""
        return {
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "context_length": self.context_length or DEFAULT_CONTEXT,
            "note": self.note,
            "tags": self.tags,
        }
