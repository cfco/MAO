# MAO · 自研多智能体协作系统 · 需求说明

> 项目：MAO（agent/ 包，入口 `mao = "agent.__main__:main"`）
> 目标：**三个臭皮匠顶个诸葛亮**——用多个免费、较慢、较弱的模型分工协作，产出比单模型更好的结果。
> 本文档描述当前已实现的架构与行为基准（2026-09-21 更新，替代描述旧 `src/mao/multi_ai_orchestrator.py` 骨架的历史版本）。

---

## 1. 核心模型

| 角色 | 说明 |
|---|---|
| 主智能体（Master） | 拆解任务、派工、汇总、把关，掌握全部本地工具（shell/文件/MCP/Skill/派工工具）。可以是池内任一 API 模型，也可以是驱动本项目的**外部智能体**（bridge 模式：谁启动驱动，谁当主） |
| 工人（Worker） | 智能体池里除主之外的成员，**纯文本执行器**（不挂本地工具）。免费模型 function calling 参差不齐，纯文本最稳最安全；本地工具权始终在主手里 |

只有 CLI/Web 等非智能体入口启动时，才需要从池里选一个当主（`--orchestrator` 或交互选择）；池为空时用 `llm:` 段兜底单模型跑 solo。

## 2. 协作模式

### 2.1 swarm（默认）：主智能体自主派工

派工能力做成主的工具，由主在 Agent Loop 里自主决定何时拆解、派工、汇总：

| 工具 | 说明 |
|---|---|
| `ask_worker` | 单个子任务派给指定工人（纯文本执行） |
| `ask_workers` | 同一子任务并行派多人，多方案对比、交叉验证；**输出按入参工人顺序排列（确定性）**，timeout 对收集阶段整体限时 |
| `ask_vote` | 两步投票取共识：先并行收集各工人方案，再让全体对**确定性编号**的方案投票，超阈值（默认 `collaboration.vote_threshold`=0.5 过半）即宣布共识并给出胜出方案全文 |
| `run_pipeline` | 固定流水线（见 2.2） |

派工基础设施（WorkerPool）：
- **LRU 缓存**：同 (工人, 任务, 角色) 不重复调用，成功回答才缓存（`collaboration.cache_size`）；缓存命中零成本，**不受节点冷却影响**（冷却只为不再打坏节点，已存的答案照用）；
- **并发去抖**：并发同 key 只打一次网络，跟随者复用发起者结果；发起者挂死时跟随者按 `collaboration.inflight_timeout` 接管重派，不无限阻塞（发起者结算时只在"自己仍是该 key 的当前在飞项"才清理 inflight，否则会误删接管者的在飞项、导致后来者重复打网络）；
- **节点冷却**：连续失败达 `collaboration.cooldown_fails` 进入指数冷却（基数 `cooldown_base`×连败次数），冷却期快速跳过；成功清零。

### 2.2 固定流水线：起草 → 评审 → 修订择优

适用高质量产出（方案/报告/代码/文案）：

1. **起草**：多名工人并行各出一稿；
2. **评审**：评审工人对全部候选稿挑错、给编号改进建议；
3. **修订择优**：修订工人各自综合评审意见出终稿，**多份终稿由工人投票择优，只返回胜出的那一份**（`select_best` 复用投票第二步；仅一份时直接采用；投票不可解析时按确定性顺序取第一份，不回退全量拼接）。

### 2.3 bridge：外部智能体当主

`python -m agent bridge`，stdin/stdout 各一行 JSON（UTF-8），启动即发 ready 事件。为驱动者提供：本地工具执行、Skill 加载与脚本执行、池内模型当纯文本工人（`ask`/`ask_many`/`ask_vote`）、完整任务执行（`run_task`）、固定流水线（`run_pipeline`）、持久会话（`new_session`/`chat`/`list_sessions`/`session_tools`/`close_session`，支持工具白名单隔离）。

## 3. 主循环与上下文

- Agent Loop：调模型 → 解析工具调用 → 执行 → 结果回灌 → 循环，上限 `llm.max_iterations`（默认 25）。
- 工具结果按**字节**截断（32KB）后回灌，防中文长结果撑爆上下文。
- 历史按 token 估算裁剪：上限 `context_length × CONTEXT_SAFETY(0.75)`，至少保留最近 `MAX_HISTORY(60)` 条。
- 单轮失败不崩：主 LLM 报错返回可行动的错误文案并发 error/final 事件。
- 协作式取消：`run(..., should_stop=fn)` 传入探针后，在每轮迭代开头（LLM 调用前）与每个工具执行前检查，命中即停止后续步骤、写入"已取消"会话记录并发 final 事件。粒度边界：在途的 LLM 请求/工具调用无法同步打断，最坏取消延迟 ≈ 一次请求或工具调用耗时（由 `llm.timeout` 兜底）。

## 4. 免费节点容错（"单节点故障不影响整体"）

| 机制 | 说明 |
|---|---|
| 超时+分类重试 | 每请求带 `llm.timeout`（默认 120s）；限流(429)/超时/5xx/连接失败指数退避重试（限流退避更长），认证/4xx 不重试；SDK 内置重试关闭，自定义退避是唯一重试源 |
| 节点冷却 | 见 2.1 |
| 畸形响应归一 | 中转站内容过滤会返回 `choices: []`：显式归类为可重试的服务端错误（原本是裸 `IndexError`，会绕过错误分类与重试编排一路冒到调用方）；模型给出空 content 且无工具调用时返回明确说明，不再静默输出空白答复 |
| 并发隔离 | 线程池并行派工，单工人失败/超时以文本如实带回，不拖垮整体 |
| 整体限时 | `ask_many`/`vote`/`collect` 用 `concurrent.futures.wait` 对收集阶段整体限时（默认 300s），超时工人标记"未完成"放弃等待，后台线程按 LLM 超时自行收尾 |

## 5. 工具与能力

- **内置工具**：`run_shell`（命令黑名单 + 注入模式拦截，优先 shell=False，cwd 限制在项目根内）、`read_file`（64KB 头部截断，不全量载入内存）、`write_file`/`append_file`（1MB 上限）、`list_dir`。文件工具路径 resolve 后必须落在项目根内，杜绝穿越。
  - **黑名单按「命令首词」匹配**：危险命令只有被当作命令执行时才拦（`format D:`、`del /f /q /s x`、`shutdown /r`），出现在参数位的同名词不拦（`ruff format .`、`make clean`、`git log --pretty=format:%H`）——早期按任意位置匹配会大面积误拦正常开发命令。匹配前先归一 `cmd /c`·`cmd /k` 包装、路径前缀、`.exe`/`.com` 后缀；并按 `&`/`|`/`;` 切段逐段查首词（`echo x & del /s y` 也拦）。PowerShell 侧拦 `-EncodedCommand` 及其合法缩写（`-e`/`-enc`/`-ec`），`-ExecutionPolicy` 不误伤。
- **MCP**：config.yaml `mcp_servers` 声明，stdio/HTTP 双传输；后台线程保长连接，**断线自动重连**（调用失败会停掉事件循环触发退避重连路径；曾连上后重连失败按退避持续重试，不是一次失败即永久放弃；首连失败仍快速失败上报）。单次连接尝试失败时逆序回滚已进入的 transport/session，防 stdio 子进程句柄泄漏。工具名 `mcp__<server>__<tool>`。
- **Skill**：`skills/<名>/SKILL.md`（frontmatter name+description），启动只注入清单省上下文，`load_skill` 按需读全文，`execute_skill_script` 跑脚本。脚本入参只接受 `scripts/` 下的裸 `.py` 文件名（拒路径分隔符/盘符/`..`），并在 resolve 后二次确认落点仍在技能目录内。
- **会话级工具白名单**：`allow_tools` 裁剪注册表（bridge `new_session` 可传），实现多会话权限隔离。

## 6. 会话持久化

- JSONL 追加写（`data/sessions/<id>.jsonl`），限流落盘：`session.flush_batch`(16) 条或 `session.flush_interval`(2.0s) 触发，`Agent.close()` 兜底 flush；CLI 参数 `--session-flush-batch/interval` 经环境变量 `MAO_SESSION_FLUSH_*` 覆盖。
- **会话 id 白名单校验**：`sanitize_session_id` 只允许字母/数字/`.`/`_`/`-`，长度 1-64，显式拒绝 `..`、路径分隔符与盘符。Web `/api/chat`、bridge `new_session` 在边界发现非法 id 直接报错（不静默改写，否则调用方手里的 id 与落盘名不一致）；`Session.__init__` 再兜底，非法值绝不拿来做文件名——否则 `session_id="../../x"` 会把 JSONL 写到 `data/sessions` 之外。
- **默认会话 id 防碰撞**：未指定 id 时用 `_default_session_id()` = 秒级时间戳 + 4 位随机后缀。只用时间戳会同秒碰撞（实测同秒三次构造拿到同一文件，历史互相穿插、cleanup 按文件误删多会话），随机后缀把同秒碰撞概率压到 16⁻⁴。
- 删除走隔离区：`Session.cleanup()` 保留最近 30 个，多余移入 `data/trash/`（7 天后物理删除），CLI/Web/bridge 启动各清一次。
- 设计选择：会话只持久化 user/assistant 文本，tool_calls/tool 中间消息不落盘（省盘省 token，过程可经事件流观测）。

## 7. 入口与界面

| 入口 | 命令 | 说明 |
|---|---|---|
| CLI 对话 | `uv run mao chat [--orchestrator 名] [--solo]` | 交互选主 |
| 单次任务 | `uv run mao run "任务" [--stream]` | --stream 输出统一事件 JSON 行 |
| 固定流水线 | `uv run mao pipeline "任务" [--draft a,b --review c --revise d]` | 不选主，直接调度工人 |
| Web | `uv run mao web` | FastAPI + SSE 流式，顶栏切换主智能体（localStorage 记忆，**切换即丢弃当前会话、下一条消息开新会话**；同一 session_id 换主时后端也会丢弃旧 Agent 按新主重建），会话空闲 30 分钟自动清理，上限 50 会话；会话创建（含 MCP 连接）在全局锁外执行，不阻塞其它请求 |
| bridge | `uv run mao bridge` | 外部智能体驱动模式 |

三入口共用统一事件 schema（`agent/core/events.py`）：`llm_call / tool_start / tool_result(preview) / stage / final / error / session`。

**stdout 纯净约定**：bridge 与 `run --stream` 的 stdout 只放 JSON 行，所有诊断/警告（配置缺失变量、非法 session_id、MCP 状态、主智能体选择）统一走 stderr——混入 stdout 会让按行 `json.loads` 的调用方直接解析失败。bridge 与 `run --stream` 模式的 stdio 在 `load_config` 前即归一为 UTF-8（Windows 重定向流默认 GBK，中文事件/配置警告不转码会炸外部 UTF-8 解码）。

**同会话并发闸门**：同一 Agent（=同一 session_id）同一时刻只允许一轮 `run`；第二条并发请求非阻塞抢锁失败即返回"会话忙"提示，不写历史、不排队（Web 允许同一会话并发提交，无闸门会交错写 session.history/落盘缓冲）。

**Web 断开即取消**：SSE 生成器为 async 生成器（同步生成器阻塞在 `queue.get()`，Starlette 无法及时传导取消）。客户端断开时生成器在 await 点收到关闭，`finally` 置位 `cancel_event`，后台轮次经 `should_stop` 在下一个边界（LLM 调用前/工具执行前）停下，不再继续消耗 token；取消粒度见 §3 的协作式取消说明。

## 8. 配置参考（config.yaml）

| 段 | 关键项 |
|---|---|
| `agents` | 智能体池，按"站"组织（一 Endpoint+Key 挂多 model）；`models` 逗号分隔，`#` 前缀临时屏蔽，`@128k` 标注上下文窗口 |
| `llm` | 兜底单模型 + temperature / max_iterations / timeout / max_retries |
| `collaboration` | max_workers / vote_threshold / cache_size / cooldown_fails / cooldown_base / inflight_timeout |
| `session` | flush_batch / flush_interval（env `MAO_SESSION_*` 优先） |
| `server` | host(127.0.0.1) / port(8000) |
| `tools` | shell_timeout |
| `mcp_servers` | MCP 接入列表 |

`${VAR}` 引用环境变量或 `.env`（缺失插值为空串并告警一次）。API key 建议只放 `.env`，勿外传 `config.yaml`。

## 9. 质量门禁

- `uv run ruff check .`：E/F/W/I/B/UP，line-length 100；用 `extend-exclude = ["data"]` 追加排除运行时产物与隔离区（**不要用 `exclude`**——那是替换语义，会顶掉 ruff 默认排除表把 `.venv`/`.git` 重新纳入扫描）。
- 解释器版本以仓库根 `.python-version` 为唯一来源（CI 用不带参数的 `uv python install` 跟随它，不写死版本号）。
- 质量门禁的守卫测试：`tests/test_config_contract.py`（`.env.example` ↔ `config.yaml` 变量名契约）、`tests/test_repo_hygiene.py`（`.gitignore` 规则真生效、ruff 用 extend-exclude）。
- `uv run pytest`：全部离线（无 API key/网络），覆盖工具注册表、配置解析、LLM 重试/退避、会话限流落盘、WorkerPool 冷却/去抖/超时接管、投票编号确定性、ask_many 整体限时、流水线择优、路径防穿越、MCP 断线重连持续重试与连接回滚、同会话并发闸门、缓存命中不受冷却、bridge stderr UTF-8、should_stop 协作式取消（迭代顶部/工具前检查点）、Web SSE 断开传导取消、默认会话 id 防碰撞等。
- GitHub Actions CI：ruff + pytest。

## 10. 安全边界

- Web 只监听 127.0.0.1，不要暴露公网；`run_shell` 等价于把本机命令权交给模型（黑名单+注入拦截兜底）。
- bridge 模式等价于把本机工具权交给驱动它的外部智能体，只给你信任的智能体用。
- 会话历史、`.env`、key 均在本地，不出本机。
