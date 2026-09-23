# MAO · 自研多智能体协作系统 · 需求说明

> 项目：MAO（agent/ 包，入口 `mao = "agent.__main__:main"`）
> 目标：**三个臭皮匠顶个诸葛亮**——用多个免费、较慢、较弱的模型分工协作，产出比单模型更好的结果。
> 本文档描述当前已实现的架构与行为基准（2026-09-23 更新，替代描述旧 `src/mao/multi_ai_orchestrator.py` 骨架的历史版本）。

---

## 1. 核心模型

| 角色 | 说明 |
|---|---|
| 主智能体（Master） | 拆解任务、派工、汇总、把关，掌握全部本地工具（shell/文件/MCP/Skill/派工工具）。统一由**驱动本项目的智能体**担任（bridge 模式：谁启动驱动，谁当主）；MAO 自身不再"选池内模型当主"（chat/run/pipeline/web 入口已移除） |
| 工人（Worker） | 智能体池里除主之外的成员，**纯文本执行器**（不挂本地工具）。免费模型 function calling 参差不齐，纯文本最稳最安全；本地工具权始终在主手里 |

主智能体一律由驱动方（外部 AI）担任，MAO 不再从池里"选主"，也不再有内部 Agent 循环或持久会话。池里的模型全部作为子 AI/工人，由外部主经 bridge 无状态调用（`ask`/`ask_many`/`ask_vote`/`run_review`）。

## 2. 协作模式（外部主经 bridge 调用子 AI）

主智能体一律是**外部驱动方**（经 bridge），它自己决定何时把活分给池里的免费子 AI。MAO 侧不再有内部 Agent 循环，只提供下列无状态派工原语（对应 bridge 指令）：

| 指令 | 说明 |
|---|---|
| `ask` | 单个子任务派给指定工人（纯文本执行，无本地工具） |
| `ask_many` | 同一子任务并行派多人，多方案对比、交叉验证；**输出按入参工人顺序排列（确定性）**；同时返回 `workers`（逐工人结构化 `ok`/`status`/`answer`/`elapsed_ms`）与 `results`（文本）；`timeout` 对收集阶段整体限时 |
| `ask_vote` | 两步投票取共识：先并行收集各工人方案，再让全体对**确定性编号**的方案投票，超阈值（默认 `collaboration.vote_threshold`=0.5 过半）即宣布共识并给出胜出方案全文。**共识比例的分母只算有效票**——废票（投了不存在编号）不进分母、不参与计票，只在结果里提示张数 |
| `run_review` | 外部主给初稿（+可选背景/验收标准），多个工人只当评审团挑错、给编号改进建议，返回结构化 `reviews` |

派工基础设施（WorkerPool）：
- **参与人数上限**：`ask_many`/`collect`/`vote` 的**缺省名单**由 `pick()` 产生——按池内确定性顺序取前 `collaboration.max_participants`（默认 5）个当前可派工人（跳过冷却与当日隔离）；显式传入的名单不受该上限约束。池子一站多模型轻易几十个，全池派发会一次打出几十个请求，而并发上限只有个位数 → 尾部批次必被整体限时丢弃、额度白烧；
- **节点冷却**：连续失败达 `collaboration.cooldown_fails` 进入指数冷却（基数 `cooldown_base`×连败次数），冷却期快速跳过；成功清零。

（历史上的 LRU 答案缓存、并发在飞去重、`run_pipeline` 固定流水线、内部 Agent 自主派工 loop 均已随「单一 bridge 路线」简化移除：外部主按需发话、命中率极低，且弱模型驱动多步工具循环不可靠。）

### 2.3 bridge：外部智能体当主（一条内核，三种外壳）

`python -m agent bridge`，stdin 每行一个 JSON 请求、stdout 每行一个 JSON 响应（UTF-8），启动即发 ready 事件。诊断/警告一律走 stderr，stdout 只放 JSON 行。为驱动者（外部主）提供：本地工具执行（`call_tool`）、Skill 加载与脚本（`load_skill`/`run_skill_script`）、池内模型当纯文本子 AI（`ask`/`ask_many` 结构化/`ask_vote`）、评审团（`run_review`）、可用性预检（`health`、`list_agents` 含能力标签）。响应 `ok` 反映真实成败：`call_tool`/`ask` 失败即 `ok:false`，派工类失败带稳定 `status` 码（`ok|error|cooldown|quarantined|missing|timeout`，全内核统一、不翻译成中文），成败判断靠字段而非解析文本前缀。

同一 `Bridge` 内核另有两种外壳，行为与逐字契约和裸协议一致（外壳只做参数拼装/透传，不复制业务逻辑）：

- **`python -m agent mcp`（MCP server，stdio）**：把 11 条指令逐一映射为 MCP 工具（`mcp>=2` 官方 SDK 的 `MCPServer`），支持 MCP client 的宿主（Claude Desktop / Cursor / Qoder 等）只需在其配置里加一段 `{"command": "uv", "args": ["run", "mao", "mcp"], "cwd": "<项目根>"}` 即接入，免写子进程驱动代码。
- **`python -m agent call <指令> [JSON]`（one-shot）**：单发一条指令、stdout 打印一行 JSON 响应即退出，供脚本/技能包在"每次调用都是新进程"的宿主里使用；JSON 参数可作位置参数或从 stdin 读一行（位置参数里的 `cmd` 优先）。退出码：0=ok:true，1=指令失败（ok:false），2=参数 JSON 坏。注意每次调用都付进程冷启动成本（配置加载；配了 `mcp_servers` 时还含 MCP 首连），高频场景仍应用长驻的 bridge/mcp 外壳。

## 3. 运行模型（MAO 侧无状态）

MAO 不再有内部 Agent 循环，也不再持久化会话——它是被外部主按需调用的**无状态执行器**：

- 每次工具/派工调用独立处理、即时返回；历史上下文由外部主自持。
- 单条 LLM 请求的失败兜底在 `llm.py`：分类重试（见 §4），终态失败以「错误：…」文本回给调用方，不抛异常、不崩协议循环。
- `ask_many`/`vote`/`collect` 的收集阶段整体限时、超时工人如实标"未完成"（见 §2/§4）。

## 4. 免费节点容错（"单节点故障不影响整体"）

| 机制 | 说明 |
|---|---|
| 超时+分类重试 | 每请求带 `llm.timeout`（默认 120s）；限流(429)/超时/5xx/连接失败指数退避重试（限流退避更长），**408 请求超时与 425 Too Early 同样归入可重试**（免费中转网关超时很常见），认证与其余 4xx 不重试；SDK 内置重试关闭，自定义退避是唯一重试源 |
| 节点冷却 | 见 2.1 |
| 畸形响应归一 | 中转站内容过滤会返回 `choices: []`：显式归类为可重试的服务端错误（原本是裸 `IndexError`，会绕过错误分类与重试编排一路冒到调用方）；模型给出空 content 且无工具调用时返回明确说明，不再静默输出空白答复 |
| 同站模型回退 | 支持多模型的中转站上：某模型发生**可重试瞬时失败**（429/5xx/超时/连接）时，`WorkerPool` 自动切换**同一中转站**（`base_url`+`api_key` 相同）的其它"当前可用"模型重试；每个模型仍保持独立（独立连接池/冷却/健康档案），回退只在同站内、**不跨站**、不重复打已试过的模型。终态错误（认证等）不回退——那类错误在整站级统一出现，回退既无用又浪费其它模型的额度。重试上限由 `collaboration.fallback_models`（默认 2，0=关闭）约束：整站挂时不会把同站几十个模型挨个试穿 |
| 并发隔离 | 线程池并行派工，单工人失败/超时以文本如实带回，不拖垮整体 |
| 整体限时 | `ask_many`/`vote`/`collect` 用 `concurrent.futures.wait` 对收集阶段整体限时；限时**按规模自适应** = `ceil(参与人数 ÷ max_workers) × llm.timeout + 30s`（调用方显式传入 `timeout` 时以传入值为准）。原先固定 300s 在「池大 + 并发低」时必然截断尾部批次（25 名 / 3 并发 = 9 批，单工人耗时 >33s 即超标）。超时工人标记"未完成"放弃等待，后台线程按 LLM 超时自行收尾 |
| 当日失败隔离 + 连续运行日下线 | `ModelHealth`（`agent/core/health.py`，档案落 `collaboration.health_file`=`data/model_health.json`）：模型出现一次**不可重试的终态失败**（认证失败、非重试类 4xx）即记当天日期，当天不再向它派工（`ask`/`ask_many`/`collect`/`vote` 全部跳过并如实提示，`status=quarantined`）；429/5xx/超时等**可重试瞬时失败**耗尽重试只进短时冷却（`status=cooldown`），不再拉黑全天——免费节点限流是常态，按旧规则一次 429 就报废整天的可用池。档案按路径**进程内共享单实例**（`health.get_health()`）：多工人池/多调用方各持独立内存账本 + 全量覆写会互相抹记录（lost update）、且彼此看不到隔离结果，共享后当日隔离全进程一致。**文件 IO 全部在锁外**（锁内更新内存取快照、锁外原子写盘），避免记账持锁把 health 锁串到 pool 锁卡住派工。某模型连续 `COLLAB_RETIRE_DAYS`（默认 7）个**运行日**（程序实际启动工作的天，未运行日子不计入不打断）都出现终态（不可重试）失败，纯瞬时 429/5xx/超时不计入、不触发下线，自动在 `model_registry.txt` 对应 `*_MODELS` 行给该模型加 `#` 下线（删 `#` 手动恢复；`LLM_MODEL` 兜底行不自动动；已下线幂等不重复改写、不刷频）。 |

## 5. 工具与能力

- **内置工具**：`run_shell`（命令黑名单 + 注入模式拦截，优先 shell=False，cwd 限制在允许的工作区内）、`read_file`（64KB 头部截断，不全量载入内存）、`write_file`/`append_file`（1MB 上限）、`list_dir`。文件工具路径 resolve 后必须落在**允许的根目录**内，杜绝穿越。
- **工作区（`tools.workspace`）**：允许的根 = MAO 项目根 + 该配置列出的外部项目目录（默认空 → 仅项目根，原有安全边界不变）。为什么要有：MAO 的定位是"谁启动谁当主、驱动项目干活"，而它经常要驱动**别的项目**（在 MAO 里分析/改造另一个仓库）；工具层若只认自己的根，主智能体连目标项目的一个文件都读不到，"驱动外部项目"就成了空话。外部项目的覆盖写同样留档到 `data/backup/`，留档名带根名前缀，避免不同项目的同名文件互相覆盖。相对路径仍一律基于项目根解析，不做含义漂移。
  - **`write_file` 覆盖前自动留档**：目标文件已存在时先复制到 `data/backup/`（`<相对路径>.时间戳.bak`，30 天惰性清理，>1MB 不留档），返回文本里给出留档路径可回滚。工具调用是同步的、等不了人工确认，所以不引入交互式确认，改为"自动留一份 + 如实告知"——覆盖是不可逆操作，至少要可恢复。
  - **黑名单按「命令首词」匹配**：危险命令只有被当作命令执行时才拦（`format D:`、`del /f /q /s x`、`shutdown /r`），出现在参数位的同名词不拦（`ruff format .`、`make clean`、`git log --pretty=format:%H`）——早期按任意位置匹配会大面积误拦正常开发命令。匹配前先归一 `cmd /c`·`cmd /k` 包装、路径前缀、`.exe`/`.com` 后缀；并按 `&`/`|`/`;` 切段逐段查首词（`echo x & del /s y` 也拦）。PowerShell 侧拦 `-EncodedCommand` 及其合法缩写（`-e`/`-enc`/`-ec`），`-ExecutionPolicy` 不误伤。
  - **注入模式匹配前先屏蔽引号内容**：`echo "a|b|c"` 里的管道符是字面量、不是命令拼接，直接对整串匹配会误拦（而 `python -c "print(1|2)"` 又因规则要求两侧都有分隔符而放行 —— 同一类写法两种结果）。现在先把成对引号内的内容替换为等长占位符再匹配：`cmd1 && cmd2`、`a | b | c` 照拦，引号外的拼接（`echo "x" && del y`）也照拦。
- **MCP**：config.yaml `mcp_servers` 声明，stdio/HTTP 双传输；后台线程保长连接，**断线自动重连**（调用失败会停掉事件循环触发退避重连路径；曾连上后重连失败按退避持续重试，不是一次失败即永久放弃；首连失败仍快速失败上报）。单次连接尝试失败时逆序回滚已进入的 transport/session，防 stdio 子进程句柄泄漏。工具名 `mcp__<server>__<tool>`。**连接按配置进程内共享**（`mcp_client.acquire_group`，签名 = `mcp_servers` 内容）：同进程只维护一套连接，**引用计数**保证最后一个使用者 `close()` 时才真正断开。
- **Skill**：`skills/<名>/SKILL.md`（frontmatter name+description），启动只注入清单省上下文，`load_skill` 按需读全文，`run_skill_script` 跑脚本。脚本入参只接受 `scripts/` 下的裸 `.py` 文件名（拒路径分隔符/盘符/`..`），并在 resolve 后二次确认落点仍在技能目录内。

## 6. 会话与持久化

MAO 侧**无状态**：不再持久化会话、不再有 JSONL 落盘 / 会话 id 校验 / 隔离区清理 / 会话级工具白名单——历史上下文全部由外部主自持。每次调用独立处理、即时返回。

## 7. 入口

只保留 **bridge 一条使用路线**：外部 AI（千问办公 / WorkBuddy 等）当主智能体，通过 `python -m agent bridge` 的 stdin/stdout JSON 行协议驱动 MAO。原先「MAO 自己选主跑」的 chat / run / pipeline / web 四个入口已移除。

| 入口 | 命令 | 说明 |
|---|---|---|
| bridge | `uv run mao bridge` | 外部智能体驱动模式（谁启动驱动，谁当主）。提供本地工具执行（`call_tool`）、Skill（`load_skill`/`run_skill_script`）、池内模型派工（`ask` / `ask_many` 并行+结构化 / `ask_vote` 投票验证 / `run_review` 评审团）、可用性预检（`health`、`list_agents`）。一问一答，无流式事件、无会话。 |

**stdout 纯净约定**：bridge 的 stdout 只放 JSON 行（每请求一行响应），所有诊断/警告（配置缺失变量、MCP 状态）统一走 stderr——混入 stdout 会让按行 `json.loads` 的调用方直接解析失败。bridge 的 stdio 在 `load_config` 前即归一为 UTF-8（Windows 重定向流默认 GBK，中文诊断不转码会炸外部 UTF-8 解码）。

## 8. 配置参考（config.yaml）

| 段 | 关键项 |
|---|---|
| `agents` | 智能体池，按"站"组织（一 Endpoint+Key 挂多 model）；`models` 逗号分隔，`#` 前缀临时屏蔽，`@128k` 标注上下文窗口，可选 `tags`（能力标签，供外部主选路）、`note` |
| `llm` | 兜底单模型 + temperature / timeout / max_retries |
| `collaboration` | max_workers（并发上限）/ max_participants（单次批量派工参与人数上限，默认 5，0=不限）/ vote_threshold / cooldown_fails / cooldown_base / health_file / retire_days（连续几个运行日出现终态失败即自动下线，默认 7；瞬时失败只冷却不下线）/ fallback_models（同站回退重试上限，默认 2，0=关闭） |
| `tools` | shell_timeout / workspace（额外允许工具访问的项目目录，默认为空=仅本项目根） |
| `mcp_servers` | MCP 接入列表 |

**配置分层**（`load_config`）：`${VAR}` 的取值优先级为 **shell 环境变量 > `.env` > `model_registry.txt`**。`config.yaml` 只留结构骨架（`${VAR}` / `${VAR:-默认}` 引用）；`.env` 承载**全部配置值**（端点/KEY/LLM 参数/协作参数，保密不进 git，AI 负责维护内容）；`model_registry.txt` 承载**全部模型名**（`NODE_A_MODELS` / `LLM_MODEL` 等，随仓库维护，git pull 即更新，它的 mtime 参与配置缓存失效判断，也是自动下线改写目标）。

**数值配置容错**：所有数值项（`max_workers`/`max_participants`/`retire_days`/`shell_timeout`/`vote_threshold`/`llm.timeout` 等）统一走 `Config.as_int()` / `as_float()`。类型写错（如 `max_participants: five`）不再抛 `ValueError` 崩启动，改为**警告一次 + 回退默认值**（提示走 stderr），与"缺 key 不阻启动、只警告"的既有态度保持一致。分层边界由双向守卫闭环：`model_registry.txt` 绝不出现 `*_KEY` 行（`test_real_registry_never_carries_key_values` 守着，真实 key 不会随 git 泄漏）；`.env` 出现 `*_MODEL(S)` 模型清单变量时重载配置即打 `[配置分层提醒]`（只报变量名绝不回显值——这些变量会盖住 registry 的 git 更新与自动下线）。`.env` 值支持 ` #` 行内注释（剥离后才是真值）。

## 9. 质量门禁

- `uv run ruff check .`：E/F/W/I/B/UP，line-length 100；用 `extend-exclude = ["data"]` 追加排除运行时产物与隔离区（**不要用 `exclude`**——那是替换语义，会顶掉 ruff 默认排除表把 `.venv`/`.git` 重新纳入扫描）。
- 解释器版本以仓库根 `.python-version` 为唯一来源（CI 用不带参数的 `uv python install` 跟随它，不写死版本号）。
- 质量门禁的守卫测试：`tests/test_config_contract.py`（`model_registry.txt` + `.env` ↔ `config.yaml` 变量名契约）、`tests/test_repo_hygiene.py`（`.gitignore` 规则真生效、ruff 用 extend-exclude）。
- `uv run pytest`：全部离线（无 API key/网络），覆盖工具注册表、配置解析、LLM 重试/退避/畸形响应归一、WorkerPool 冷却/当日隔离、ask_many 整体限时与确定性顺序、结构化派工结果（`ask_many_structured`）、`health` 可用性探针、能力标签 `tags`、`run_review` 评审团、路径防穿越（工具/技能脚本）、MCP 断线重连持续重试与连接回滚、bridge stderr UTF-8、配置分层（model_registry.txt 模型名基础层 / .env 全配置覆盖层 / shell 最高 / registry 与 .env 变更失效缓存 / 行内注释剥离 / 边界双向守卫：registry 绝不带 `*_KEY`、`.env` 混入 `*_MODEL(S)` 按名告警不回显值）、模型健康（当日隔离、进程内共享、幂等、跨进程持久化、连续运行日自动下线含"未运行的日子不计入也不打断"；`tests/conftest.py` 用 `MAO_HEALTH_FILE` / `MAO_MODEL_REGISTRY_FILE` 给每用例独立档案与独立 registry 防跨用例污染）等。（内部 Agent 主循环、会话持久化、LRU 缓存、并发去抖、`run_pipeline` 的用例已随功能移除。）
- 2026-09-22 审计修复的专项回归集中在 `tests/test_audit_fixes.py`：健康档案进程内共享（含"两个实例各自先读盘、再交错记账不得互相覆盖"的 lost update 场景）且直连构造仍独立、缺省派工规模上限（pick）与自适应限时、显式名单不被上限约束、批量路径跳过项如实带回、MCP 连接共享与引用计数、计票不采信错误文本。**配套变异测试**（撤销修复 → 对应用例必须失败）确认用例非恒真。
- 2026-09-22 第二轮全链路审计的回归集中在 `tests/test_audit2_fixes.py`（27 例）：投票分母只算有效票、健康档案写盘不阻塞隔离查询与并发落盘不留垃圾、6 项数值配置的类型容错、流水线阶段间裁剪（含窗口伸缩与按份均分）、`write_file` 覆盖留档、注入检查不误拦引号内字面量且真实拼接照拦、408/425/429/5xx 与 4xx 的重试分类。变异校验扩到多项（`tests/mutation_check.py`）；本轮变异又抓出 2 个恒真用例并修正（假 `_last_active` 被清理线程连带清理、测试 monkeypatch 掉了被测方法本身）。
- GitHub Actions CI：ruff + pytest。

## 10. 安全边界

- MAO 不再对外开任何网络端口（Web 入口已移除）；`run_shell` 等价于把本机命令权交给主智能体（黑名单+注入拦截兜底）。
- bridge 通过本地 stdin/stdout 通信，等价于把本机工具权交给驱动它的外部智能体，只给你信任的智能体用。
- `.env`、key、健康档案均在本地，不出本机（MAO 不持久化会话历史）。
