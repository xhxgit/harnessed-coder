# Harnessed Coder

一个本地运行、支持多种扩展能力的交互式 Coding Agent，支持多 workspace/session、长上下文自动压缩、分级权限审批、用户级和工作区级长期记忆、工具按需发现、Skills、`AGENTS.md`、MCP、子代理，以及运行日志、JSONL trace 和 token 用量统计。

## 快速开始

运行要求（当前仅支持 Windows，暂不支持 Linux）：

- Windows 10/11
- Python 3.14 或更高版本
- [uv](https://docs.astral.sh/uv/)

安装依赖：

```powershell
uv sync
```

启动 Agent，并指定允许操作的工作区：

```powershell
uv run harnessed-coder --root C:\path\to\your-project
```

首次运行会创建 `~/.harnessed-coder/config.json` 并退出。填写真实 API Key 后，再次执行启动命令即可进入交互界面。

如需指定本地数据目录：

```powershell
uv run harnessed-coder `
  --root C:\path\to\your-project `
  --data-dir C:\path\to\agent-data
```

## 配置

默认配置如下：

```json
{
  "_comment": "Automatic context compression starts at 240000 tokens (80% of context_max_tokens). Replace openai_api_key before the first model request.",
  "openai_api_key": "sk-",
  "openai_base_url": "https://api.deepseek.com",
  "model": "deepseek-flash",
  "context_max_tokens": 300000,
  "responses_full_history": true,
  "reasoning_effort": "low",
  "version": 1
}
```

主要字段：

- `openai_api_key`：OpenAI-compatible API Key
- `openai_base_url`：API Base URL；留空或删除时使用 OpenAI SDK 默认地址
- `model`：主模型名称
- `context_max_tokens`：会话历史规划预算
- `responses_full_history`：是否为 Responses API 重发完整历史
- `reasoning_effort` / `thinking_budget`：可选推理配置

`context_max_tokens` 不是模型完整上下文窗口的硬上限。自动压缩水位由项目常量计算，当前为该值的 80%。对于不支持 `previous_response_id` 状态复用的 Responses 实现，应启用 `responses_full_history`。

## 使用

直接输入编码任务即可。输入 `/help` 查看完整命令列表。

| 命令 | 说明 |
|---|---|
| `/workspace ...` | 查看、列出或切换工作区 |
| `/session ...` | 查看、切换、重命名或清空会话 |
| `/model [name]` | 查看或切换模型 |
| `/status` | 查看当前运行状态 |
| `/tokens` | 查看上下文和 API token 用量 |
| `/history search <query>` | 搜索当前会话 |
| `/memory ...` | 管理长期记忆 |
| `/skills [reload]` | 查看或重新加载 Skills |
| `/compact` | 手动压缩当前上下文 |
| `/exit` | 退出 CLI |

常用启动参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--root` | 当前目录 | Agent 工作区根目录 |
| `--data-dir` | `~/.harnessed-coder` | 配置、会话、日志和记忆目录 |
| `--workspace` | 从 root 派生 | workspace 数据槽名称 |
| `--session` | `default` | 初始 session |
| `--api-log-dir` | 禁用 | 保存完整模型请求与响应 |
| `--auto-memory-extraction` | 未启用 | 开启成功轮次后的自动记忆提取 |
| `--no-tool-batch-summary` | 未启用 | 关闭工具批次摘要 |

```powershell
uv run harnessed-coder --help
```

## Skills 与项目指令

Skills 的默认位置：

```text
~/.harnessed-coder/skills/<skill-name>/SKILL.md
<workspace>/.harnessed-coder/skills/<skill-name>/SKILL.md
```

同名 Skill 会同时保留；存在歧义时使用 `user:<name>` 或 `workspace:<name>`。工作区根目录的 `AGENTS.md` 会作为项目指令加载。

## MCP

在 `config.json` 中增加 `mcp_servers` 即可连接 stdio 或 Streamable HTTP 服务：

```json
{
  "mcp_servers": {
    "local": {
      "transport": "stdio",
      "command": "uvx",
      "args": ["some-mcp-server"]
    },
    "remote": {
      "transport": "streamable_http",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "${MCP_TOKEN}"
      }
    }
  }
}
```

可以使用 `include` 或 `exclude` 精确过滤工具。MCP 工具会进入现有权限确认流程；图片、音频和 blob 会先保存到本地 artifact store，不会自动写入工作区。

## 数据与安全

配置、会话、记忆、日志、trace 和 MCP artifacts 默认保存在 `~/.harnessed-coder`。使用 `--data-dir` 可以更改位置。

文件工具受 `--root` 限制，PowerShell 调用经过策略判断、LLM reviewer 或人工确认。但这不是操作系统级沙箱。运行未知任务时，请使用可恢复的仓库、独立分支或临时工作区。

日志和 trace 可能包含提示词、文件内容、命令或模型输出。完整 API 日志默认关闭；启用 `--api-log-dir` 后应将输出视为敏感数据。
