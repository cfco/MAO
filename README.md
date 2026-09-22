# MAO - 自研多智能体协作系统

**目标：三个臭皮匠顶个诸葛亮**——用多个免费、较慢、较弱的模型分工协作，产出比单模型更好的结果。

- **外部 AI 当主，MAO 只当执行器**：本项目通过 bridge 被外部智能体（千问办公 / WorkBuddy 等）驱动，驱动者即主智能体。MAO 不再自带"选主自己跑"的入口（chat / run / pipeline / web 已移除），只提供本机工具（手）与池内免费模型（子 AI）。
- 主智能体（外部 AI）：拆解任务、派工、汇总、把关，通过 bridge 调用 MAO 的全部本地工具（shell / 文件 / MCP / Skill）。
- 子智能体（工人）：池里的 API 模型，纯文本执行器，负责并行干子任务、提供第二意见、交叉验证（`ask` / `ask_many` / `ask_vote` / `run_review`）。

## 目录结构

```
MAO/
├── agent/
│   ├── __main__.py         # CLI 入口（仅 bridge 一条路线）
│   ├── bridge.py           # 外部智能体驱动协议（谁驱动谁当主）
│   ├── config.py           # 智能体池 AgentProfile + 协作配置
│   ├── skills_manager.py   # Skill 机制
│   ├── core/
│   │   ├── llm.py          # LLM 适配层（OpenAI 兼容协议）
│   │   ├── orchestrator.py # 工人池：ask / ask_many(结构化) 并行派工 + ask_vote 投票验证 + health 预检
│   │   └── health.py       # ModelHealth：当日失败隔离档案（data/model_health.json）
│   └── tools/
│       ├── base.py         # 工具基类 + 注册表
│       ├── builtin.py      # 内置工具（shell/文件）
│       └── mcp_client.py   # MCP 接入层（stdio/HTTP）
├── skills/                 # 技能目录（example_hello 是示例）
├── config.yaml             # 智能体池 / MCP / 工具配置
└── docs/架构方案.md
```

更多文档：
| 文档 | 说明 |
|------|------|
| [docs/架构方案.md](docs/架构方案.md) | 系统整体架构设计 |
| [docs/外部主接入指南.md](docs/外部主接入指南.md) | 外部智能体（TRAE 等）通过 bridge 协议当主智能体的接入教程 |

## 快速开始

推荐用 [uv](https://github.com/astral-sh/uv) 管理环境与依赖（快、可锁定版本、无需手动 activate）。

### 使用 uv（推荐）

1. 安装 uv（一次）：`pip install uv`，详见官方文档。
2. 同步依赖并创建虚拟环境：

   ```
   uv sync            # 读取 pyproject.toml，创建 .venv 并安装全部依赖
   ```

   > 国内加速：把 PyPI 镜像设为清华源。新建用户级配置
   > `C:\Users\<你>\AppData\Roaming\uv\uv.toml`（macOS/Linux 为 `~/.config/uv/uv.toml`）：
   > ```toml
   > [[index]]
   > url = "https://pypi.tuna.tsinghua.edu.cn/simple"
   > default = true
   > ```

3. 运行（入口命令 `mao`，由 `uv run` 驱动）：

   ```
   uv run mao bridge     # 外部智能体驱动模式（谁启动驱动，谁当主智能体）
   ```

   > 本项目只保留 **bridge 一条使用内核**（外加两个壳）：外部 AI 当主智能体，借用 MAO
   > 的"手"（run_shell / 文件 / MCP / Skill）与"子 AI"（`ask` / `ask_many` 并行派工、
   > `ask_vote` 投票验证、`run_review` 评审团）。接入按省事程度依次是
   > **MCP（`mao mcp`，宿主填一段配置即用）**、**one-shot（`mao call`，脚本一行命令）**、
   > **裸协议（`mao bridge`，自管子进程逐行 JSON）**。原先的 chat / run / pipeline / web
   > 「MAO 自己选主跑」入口已移除。接入步骤见
   > [docs/外部主接入指南.md](docs/外部主接入指南.md)。

### 使用 pip（备选）

1. 安装依赖（Python 3.10+，依赖清单以 `pyproject.toml` 为准）：

   ```
   pip install .
   ```

2. 运行：

   ```
   python -m agent bridge     # 外部智能体驱动模式（谁启动驱动，谁当主智能体）
   ```

### 配置智能体池

**第 1 步：填 key（写进 `.env`，不进 git）**

接口地址与模型清单已经直接写在 `.env.example` 里（bynara + kilo 免费站，随仓库维护，
`git pull` 即更新模型）。你只需要在项目根目录建一个 `.env`，**只写 key**：

```ini
NODE_A_KEY=sk-你的bynara-key
NODE_B_KEY=sk-你的kilo-key    # 不用 kilo 可不写
LLM_API_KEY=sk-同一个bynara-key  # 兜底单模型（池里节点全挂时用）
```

加载优先级：shell 环境变量 > `.env` > `.env.example`。要改接口/增删模型改
`.env.example`（或设环境变量）即可，不必碰 `.env`。变量名必须与 `config.yaml`
里的 `${...}` 一致，否则填了不生效（`tests/test_config_contract.py` 会守住这条契约）。

两层互为守卫，各管各的边界：`.env.example`（AI 可改、进 git）里的 `*_KEY` 行
永远必须是空占位（有测试守着，真实 key 不会被误提交）；反过来 `.env` 里若混进
非 key 变量（如复制了整份 `.env.example`），启动时按 `[配置分层提醒]` 点名警告
（只报变量名、绝不回显值）——因为这些变量会盖住 `.env.example` 的模型清单更新。

模型会自动体检：某模型出现一次**不可重试的终态失败**（如认证被拒）→ 当天不再派工给它
（当日隔离）；429/5xx/超时等瞬时失败耗尽重试只进短时冷却——免费 API 的限流抖动是常态，
不该一次 429 就报废整天可用池。已移除"连续多日失败自动改 `.env.example` 下线"——
按需驱动下路由取舍交给外部主，可用 `health` 指令查各节点当前是否可用。

**第 2 步：配池（编辑 `config.yaml` 的 `agents`）**

按"站"组织：一个 Endpoint + 一个 Key 可以挂多个模型，每个模型自动成为一个智能体。
可选 `tags`（逗号分隔的能力标签）帮外部主决定"这类活派给谁"——`health`/`list_agents` 会把标签回吐给主。

```yaml
agents:
  - name: node-a
    base_url: ${NODE_A_ENDPOINT}
    api_key: ${NODE_A_KEY}
    models: ${NODE_A_MODELS}
    note: 中转站A
    tags: code,长上下文      # 可选：给外部主选路用的能力标签
```

**第 3 步：验证**

```
# 起 bridge，发一行 list_agents，应回出池内智能体（不含 key）
echo '{"cmd":"list_agents"}' | uv run mao bridge
```

免 key 方案：本地跑 Ollama（`base_url: http://localhost:11434/v1`，`api_key: ""`）。池为空时会用 `llm:` 段的兜底单模型跑 solo（`llm:` 段的 `base_url`/`model` 带默认值，只有 `LLM_API_KEY` 必填）。

## 多智能体协作（核心玩法）

外部主 AI 通过 bridge 指令直接调度池里的免费模型当"子 AI"，自己决定何时并行、何时验证：

- `ask`：把子任务派给某个工人（纯文本执行，无本地工具）；
- `ask_many`：同一任务并行派多个工人，结果按入参顺序返回，含每个工人的 `ok`/耗时/状态（结构化 `workers` + 文本 `results`），适合多方案对比、交叉验证；
- `ask_vote`：两步投票取共识（先各出方案，再对编号投票）；
- `run_review`：外部主给初稿，多个工人只当评审团挑错。

工人是纯文本执行器（不挂本地工具）：免费模型 function calling 参差不齐，这样最稳也最安全，本地工具权始终在主智能体（外部 AI）手里。

### 外部智能体驱动（谁启动谁当主）：一条内核，三种外壳

任何能起子进程、能读写文本行的智能体都能当主。`bridge` 是唯一内核（11 条无状态指令，
下表），外面有两种更省事的壳：

| 接入方式 | 适合谁 | 上手成本 |
|---|---|---|
| **`uv run mao mcp`（推荐）** | Claude Desktop / Cursor / Qoder 等支持 MCP 的宿主 | 宿主配置加一段，工具自动发现，协议零感知 |
| **`uv run mao call <指令> '<JSON>'`** | 只有 shell/技能机制、每次调用起新进程的宿主（脚本、SKILL.md） | 一行命令拿一行 JSON，退出码表成败 |
| **`uv run mao bridge`（裸协议）** | 要长驻、批量、自己管进出的深度集成驱动方 | 自写子进程驱动（逐行 JSON） |

**MCP 接入**（宿主配置里加一段即可，11 条指令逐一映射为 MCP 工具）：

```json
{
  "mcpServers": {
    "mao": { "command": "uv", "args": ["run", "mao", "mcp"], "cwd": "D:/path/to/MAO" }
  }
}
```

**one-shot 接入**（脚本里最省事的一条；JSON 也可从 stdin 读）：

```
uv run mao call health
uv run mao call ask '{"agent":"glm-flash","prompt":"总结一下这个仓库的结构"}'
uv run mao call call_tool '{"name":"run_shell","args":{"command":"git status --short"}}'
```

退出码：0=成功，1=指令失败（响应里带 error），2=参数 JSON 坏。

**bridge 裸协议**：

```
python -m agent bridge
```

协议：stdin 每行一个 JSON 请求，stdout 每行一个 JSON 响应（UTF-8），启动即发一行 ready 事件。
**stdout 只放 JSON 行**（诊断/警告一律走 stderr），可直接按行 `json.loads` 解析。
stderr 也已在 bridge 启动时归一为 UTF-8：Windows 重定向流默认本地代码页（GBK），
配置警告若按 GBK 写入，外部驱动方按 UTF-8 解码会直接 `UnicodeDecodeError`。

| 指令 | 说明 |
|------|------|
| `{"cmd":"ping"}` | 握手，返回版本与池内智能体名 |
| `{"cmd":"list_agents"}` | 智能体池清单（不含 key，含能力标签 tags） |
| `{"cmd":"health"}` | 各子 AI 当前可用性快照（冷却剩余/当日隔离/标签），派工前预检 |
| `{"cmd":"list_tools"}` | 本地工具清单（内置+MCP+Skill） |
| `{"cmd":"call_tool","name":"run_shell","args":{...}}` | 执行本地工具 |
| `{"cmd":"load_skill","name":"example_hello"}` | 读技能全文 |
| `{"cmd":"run_skill_script","skill":"...","script":"...","args":{}}` | 跑技能脚本 |
| `{"cmd":"ask","agent":"glm-flash","prompt":"..."}` | 把池内模型当纯文本大脑用 |
| `{"cmd":"ask_many","workers":["a","b"],"prompt":"..."}` | 同一任务并行派多个工人；返回 `workers`（结构化逐工人 ok/status/answer/elapsed_ms）+ `results`（文本） |
| `{"cmd":"ask_vote","prompt":"...","threshold":0.5}` | 两步投票取共识 |
| `{"cmd":"run_review","draft":"...","context":"可选"}` | 外部主给初稿，子 AI 只当评审团挑错（返回结构化 `reviews`） |

外部智能体自己的大脑在它那边；本项目给它"手"（工具/技能）和"工人"（池内模型）。

## 用 MAO 驱动别的项目（工作区）

MAO 默认只能读写**自己的**项目根。要让它去分析/改造**另一个仓库**，在 `config.yaml` 里登记那个目录：

```yaml
tools:
  workspace:
    - ../other-proj           # 相对路径基于 MAO 项目根
    - D:/work/other-proj      # 绝对路径也行
```

配好之后主智能体就能读目标项目的文件，也能把 `run_shell` 的工作目录指过去。外部驱动方（如 WorkBuddy）通过 bridge 调用时同样生效：

```json
{"cmd":"call_tool","name":"read_file","args":{"path":"D:/work/other-proj/README.md"}}
{"cmd":"call_tool","name":"run_shell","args":{"command":"python -m pytest -q","cwd":"D:/work/other-proj"}}
```

要点：

- **默认为空 = 仍然只能碰项目根**，安全边界不变；配了就等于把那些目录的读写权一并交给主智能体，只加自己信任的项目。
- 覆盖外部项目已有文件时，原文件会自动留档到 `data/backup/`，留档名带根名前缀 —— 不同项目的同名文件（如两个仓库都有 `README.md`）不会在留档目录里互相覆盖。
- 相对路径**仍然**一律基于 MAO 项目根解析，加工作区不会改变相对路径的含义。
- 在允许根之外的路径（例如 `C:/Windows/win.ini`）依旧被拒绝，越界防护没有因为这项放宽而消失。

## 接入 MCP

`config.yaml` 的 `mcp_servers` 加一段即可，启动时自动连接、发现工具，工具名格式 `mcp__<server>__<工具>`：

```yaml
mcp_servers:
  - name: filesystem
    transport: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "D:/work/demo-dir"]
  - name: remote
    transport: http
    url: https://mcp.example.com/mcp
```

## 写一个新技能

1. `skills/` 下建文件夹，写 `SKILL.md`（frontmatter 的 `name` + `description` 必填，description 决定主智能体什么时候想到用它）
2. 需要脚本就放 `scripts/`，脚本从 `argv[1]` 接收 JSON 参数，结果 print 到 stdout
3. 重启生效；运行时 Agent 先 `load_skill` 读全文，再用 `run_skill_script` 跑脚本

参考示例：`skills/example_hello/`

## 安全说明

- 内置 `run_shell` 可直接执行本机命令，等于把本机命令权交给主智能体；bridge 通过本地 stdin/stdout 通信，不监听任何网络端口
- `bridge` 模式等价于把本机工具权交给驱动它的外部智能体，只给你信任的智能体用
- ⚠ **`mao mcp` 把接入门槛降到"宿主配置里加一行"**：任何连上它的 MCP 宿主（Claude Desktop / Cursor 等）都能经 `call_tool → run_shell` 在你这台机器执行命令，而 `run_shell` 是命令卫生黑名单、**并非沙箱**（`python -c` 之类可绕，见 `builtin.py` 说明）。接入前确认信任该宿主；生产环境建议用容器 / 独立账户隔离，`tools.workspace` 只配必要目录、默认为空。
- `config.yaml` 里若直接填 API key，注意不要外传该文件
