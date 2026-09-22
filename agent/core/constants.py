"""核心常量：派工结果状态码。

抽离原因：VotingMixin（voting.py）和 WorkerPool（orchestrator.py）都要用到这些
状态码，直接互相 import 会循环依赖。独立成文件后两边都从这里 import，避免循环。

这些状态码是对外稳定契约（ask_many_structured / ask_result 的 status 字段）：
新增/修改必须同步改对应的 bridge/mcp 三外壳处理代码与测试。
"""
from __future__ import annotations

# ---------- 派工结果状态 ----------
# 成败由**控制流**决定（是否跳过、chat 是否抛错），绝不从工人回答文本里猜：
# 工人正常回答若以「错误：」开头也不会被误判成失败。
ST_OK = "ok"                    # 成功拿到回答
ST_ERROR = "error"              # 打了网络但失败（LLMError / 意外异常 / 取 client 失败）
ST_COOLDOWN = "cooldown"        # 冷却中，本轮跳过
ST_QUARANTINED = "quarantined"  # 当日失败隔离，本轮跳过
ST_MISSING = "missing"          # 指定工人不在池里
ST_TIMEOUT = "timeout"          # 批量派工整体限时内未回来（由收集层标注）
# 以上状态码即对外 status 词表（ask_result / ask_many_structured / 单 ask 统一使用），
# 不再另设中文标签翻译层：外部主按稳定英文码程序化消费。

# 两条系统提示词：派工用的普通系统提示 + 投票用的评审提示
WORKER_SYSTEM = (
    "你是协作团队中的工人智能体。认真完成分配给你的子任务，"
    "直接给出结果内容本身，不要客套、不要复述任务。"
    "如果子任务无法完成，明确说明原因和你的困难。"
)
VOTER_SYSTEM = (
    "你是协作团队中的评审工人。下面给你一份方案清单，"
    "请选出你认为最好的那份，只输出它的编号（一个数字），不要输出任何其他内容。"
)
