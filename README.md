# MAO - 自研多智能体协作系统

**目标：三个臭皮匠顶个诸葛亮**——用多个免费、较慢、较弱的模型分工协作，产出比单模型更好的结果。

- **谁启动，谁是主**：用外部智能体启动/驱动本项目时，它自动成为主智能体；CLI/Web 启动时才需要从 API 智能体池里选一个当主。
- 主智能体：拆解任务、派工、汇总、把关，掌握全部本地工具（shell / 文件 / MCP / Skill）。
- 子智能体（工人）：池里其余的 API 模型，纯文本执行器，负责并行干子任务、提供第二意见、交叉验证。

## 目录结构

```
MAO/
├── agent/
│   ├── __main__.py         # CLI 入口（bridge / chat / run / web）
│   ├── bridge.py           # 外部智能体驱动协议（谁驱动谁当主）
│   ├── config.py           # 智能体池 AgentProfile + 协作配置
│   ├── skills_manager.py   # Skill 机制
│   ├── core/
│   │   ├── agent.py        # 主智能体 Loop（支持注入主 profile + 工人工具）
│   │   ├── llm.py          # LLM 适配层（OpenAI 兼容协议）
│   │   ├── orchestrator.py # 工人池：ask_worker / ask_workers 并行派工
│   │   ├── pipeline.py     # 固定流水线：起草→评审→修订择优
│   │   ├── events.py       # 统一事件 schema（CLI/Web/bridge 三入口共用）
│   │   └── session.py      # 会话与历史落盘
│   └── tools/
│       ├── base.py         # 工具基类 + 注册表
│       ├── builtin.py      # 内置工具（shell/文件）
│       └── mcp_client.py   # MCP 接入层（stdio/HTTP）
├── web/                    # Web 服务 + 前端（顶栏可选主智能体）
├── skills/                 # 技能目录（example_hello 是示例）
├── data/sessions/          # 会话历史
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
   uv run mao chat                    # 交互对话；池里多个智能体时让你选主
   uv run mao chat --orchestrator kimi   # 直接指定主智能体
   uv run mao run "任务" --solo       # 单次任务、不启用协作
   uv run mao pipeline "任务"         # 固定流水线：起草→评审→修订择优
   uv run mao web                     # Web 界面，顶栏切换主智能体；客户端断开即取消后台轮次
   ```

### 使用 pip（备选）

1. 安装依赖（Python 3.10+）：

   ```
   pip install -r requirements.txt
   ```

2. 运行：

   ```
   python -m agent chat                    # 交互对话；池里多个智能体时让你选主
   python -m agent chat --orchestrator kimi   # 直接指定主智能体
   python -m agent run "任务" --solo       # 单次任务、不启用协作
   python -m agent pipeline "任务"         # 固定流水线：起草→评审→修订择优
   python -m agent web                     # Web 界面，顶栏切换主智能体；客户端断开即取消后台轮次
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
（只报变量名、绝不回显值）——因为这些变量会盖住 `.env.example` 的模型清单更新
和自动下线效果。模型健康下线只会改写 `.env.example` 的 `*_MODELS` 行，永不触碰
任何 `*_KEY` 行。

模型会自动体检：某模型一次请求终态失败 → 当天不再派工给它；连续 7 个"运行日"
（程序实际启动过的天，周末没开机不计入也不打断）都失败 →
自动在 `.env.example` 对应模型前加 `#` 下线（stderr 会提示；删掉 `#` 即恢复）。
阈值可调：`config.yaml` 的 `collaboration.retire_days`。

**第 2 步：配池（编辑 `config.yaml` 的 `agents`）**

按"站"组织：一个 Endpoint + 一个 Key 可以挂多个模型，每个模型自动成为一个智能体。

```yaml
agents:
  - name: node-a
    base_url: ${NODE_A_ENDPOINT}
    api_key: ${NODE_A_KEY}
    models: ${NODE_A_MODELS}
    note: 中转站A
```

**第 3 步：验证**

```
uv run mao chat        # 池里多个智能体时会让你选一个当主
```

免 key 方案：本地跑 Ollama（`base_url: http://localhost:11434/v1`，`api_key: ""`）。池为空时会用 `llm:` 段的兜底单模型跑 solo（`llm:` 段的 `base_url`/`model` 带默认值，只有 `LLM_API_KEY` 必填）。

## 多智能体协作（核心玩法）

### swarm 模式（默认）

主智能体获得两个派工工具，自主决定何时协作：

- `ask_worker`：把独立子任务派给某个工人并行干；
- `ask_workers`：同一关键问题同时派给多个工人，多方案对比、交叉验证、投票取共识。

对话中你能实时看到派工过程（CLI 打印 / Web 卡片展示）。工人是纯文本执行器（不挂本地工具）：免费模型 function calling 支持参差不齐，这样最稳也最安全，本地工具权始终在主智能体手里。

### 外部智能体驱动（谁启动谁当主）

任何能起子进程、能读写文本行的智能体都能当主：

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
| `{"cmd":"list_agents"}` | 智能体池清单（不含 key） |
| `{"cmd":"list_tools"}` | 本地工具清单（内置+MCP+Skill） |
| `{"cmd":"call_tool","name":"run_shell","args":{...}}` | 执行本地工具 |
| `{"cmd":"load_skill","name":"example_hello"}` | 读技能全文 |
| `{"cmd":"run_skill_script","skill":"...","script":"...","args":{}}` | 跑技能脚本 |
| `{"cmd":"ask","agent":"glm-flash","prompt":"..."}` | 把池内模型当纯文本大脑用 |
| `{"cmd":"ask_many","workers":["a","b"],"prompt":"..."}` | 同一任务并行派多个工人 |
| `{"cmd":"ask_vote","prompt":"...","threshold":0.5}` | 两步投票取共识 |
| `{"cmd":"run_task","task":"...","orchestrator":"可选","solo":false}` | 完整跑一轮多智能体任务 |
| `{"cmd":"run_pipeline","task":"..."}` | 固定流水线：起草→评审→修订择优 |
| `{"cmd":"new_session","session_id":"可选","tools":["白名单"]}` | 建持久会话（可做工具隔离） |
| `{"cmd":"chat","message":"...","session_id":"..."}` | 用会话跑一轮，可续历史 |
| `{"cmd":"list_sessions"}` / `{"cmd":"session_tools","session_id":"..."}` | 查会话清单 / 某会话工具白名单 |
| `{"cmd":"close_session","session_id":"..."}` | 关会话 |

外部智能体自己的大脑在它那边；本项目给它"手"（工具/技能）和"工人"（池内模型）。

## 接入 MCP

`config.yaml` 的 `mcp_servers` 加一段即可，启动时自动连接、发现工具，工具名格式 `mcp__<server>__<工具>`：

```yaml
mcp_servers:
  - name: filesystem
    transport: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "C:/Users/Administrator/Desktop"]
  - name: remote
    transport: http
    url: https://mcp.example.com/mcp
```

## 写一个新技能

1. `skills/` 下建文件夹，写 `SKILL.md`（frontmatter 的 `name` + `description` 必填，description 决定主智能体什么时候想到用它）
2. 需要脚本就放 `scripts/`，脚本从 `argv[1]` 接收 JSON 参数，结果 print 到 stdout
3. 重启生效；运行时 Agent 先 `load_skill` 读全文，再用 `execute_skill_script` 跑脚本

参考示例：`skills/example_hello/`

## 安全说明

- 内置 `run_shell` 可直接执行本机命令，Web 只监听 127.0.0.1，不要暴露到公网
- `bridge` 模式等价于把本机工具权交给驱动它的外部智能体，只给你信任的智能体用
- `config.yaml` 里若直接填 API key，注意不要外传该文件
