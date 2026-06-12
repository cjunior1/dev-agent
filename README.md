# dev-agent

Software development assistant CLI powered by **LangGraph Deep Agent** — a ReAct loop with harness techniques: checkpointing, interrupt hooks, streaming events, and human-in-the-loop support.

## Features

- **Deep Agent loop** — ReAct (Reason + Act) with configurable max-iterations guard
- **12 development tools** — shell, filesystem, git, linting, testing, web fetch
- **Interactive REPL** — Rich-rendered streaming output with `/slash` commands
- **Webhook server** — FastAPI endpoint for GitHub and GitLab CI/CD events
- **Harness techniques** — MemorySaver checkpointing, interrupt_before/after hooks, async streaming

## Requirements

- Python 3.12+
- [Anthropic API key](https://console.anthropic.com/)

## Installation

```bash
git clone https://github.com/cjunior1/dev-agent.git
cd dev-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Create a `.env` file:

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### Single-shot prompt

```bash
dev-agent run "explain the structure of this project"
dev-agent run "find all TODO comments and summarise them" --workspace /path/to/project
```

### Interactive REPL

```bash
dev-agent chat
dev-agent chat --workspace /path/to/project
```

Available slash commands inside the REPL:

| Command    | Description                        |
|------------|------------------------------------|
| `/help`    | List slash commands                |
| `/tools`   | List all available tools           |
| `/history` | Show recent prompt history         |
| `/thread`  | Show current thread ID             |
| `/threads` | List all active threads            |
| `/new`     | Start a new conversation thread    |
| `/clear`   | Clear the screen                   |
| `/exit`    | Exit the REPL                      |

### Webhook server

```bash
dev-agent serve --port 8080 --secret my-hmac-secret --workspace /path/to/project
```

Endpoints:

| Method | Path               | Description                              |
|--------|--------------------|------------------------------------------|
| GET    | `/health`          | Health check                             |
| POST   | `/webhook/github`  | GitHub events (PR, push, issue)          |
| POST   | `/webhook/gitlab`  | GitLab events (MR, push, issue)          |
| POST   | `/webhook/run`     | Generic: `{"prompt": "...", "workspace": "..."}` |

### Config and tools

```bash
dev-agent config show      # inspect active configuration
dev-agent tools            # list all available tools
```

## Architecture

```
src/dev_agent/
├── config.py              # pydantic-settings + YAML config
├── tools/
│   ├── shell.py           # async shell execution with safety blocklist
│   ├── filesystem.py      # file_read / file_write / file_list / code_search
│   ├── git_tools.py       # git status / diff / commit / branch
│   ├── code_tools.py      # ruff/eslint lint + pytest/jest/vitest/go test
│   ├── web_tools.py       # httpx web fetch for documentation lookup
│   └── registry.py        # build_toolset() — tool factory
├── agent/
│   ├── state.py           # AgentState TypedDict
│   ├── prompts.py         # system prompt template
│   ├── graph.py           # LangGraph StateGraph (ReAct loop)
│   └── harness.py         # AgentHarness: checkpointer, hooks, streaming
├── cli/
│   ├── main.py            # Typer app: run / chat / serve / config / tools
│   └── repl.py            # Rich interactive REPL
└── webhooks/
    ├── handlers.py        # GitHub/GitLab payload → agent prompt
    └── server.py          # FastAPI webhook server
```

### Harness techniques

| Technique              | Implementation                                              |
|------------------------|-------------------------------------------------------------|
| Checkpointing          | `MemorySaver` — per-thread conversation memory              |
| Interrupt hooks        | `interrupt_before` / `interrupt_after` in `settings.yaml`  |
| Streaming              | `astream_events` v2 — token / tool_call / tool_result / done |
| Max iterations guard   | Stops the ReAct loop at `agent.max_iterations` tool calls   |
| Human-in-the-loop      | `AgentHarness.resume(thread_id)` to continue after interrupt |

## Configuration

Edit `config/settings.yaml` or use environment variables (prefix `DEV_AGENT_`):

```yaml
agent:
  model: "claude-sonnet-4-6"   # Anthropic model ID
  max_iterations: 25            # max tool calls per run
  temperature: 0.1
  streaming: true

harness:
  checkpointing: true
  interrupt_before: []          # node names to pause before
  interrupt_after: []           # node names to pause after

webhooks:
  secret: ""                    # HMAC-SHA256 secret for GitHub
  port: 8080
```

Environment variable examples:

```bash
DEV_AGENT_AGENT__MODEL=claude-opus-4-8
DEV_AGENT_AGENT__MAX_ITERATIONS=50
DEV_AGENT_WEBHOOKS__SECRET=my-secret
```

## Available Tools

| Tool          | Description                                          |
|---------------|------------------------------------------------------|
| `shell`       | Run shell commands (bash) with safety blocklist      |
| `file_read`   | Read file contents with line numbers                 |
| `file_write`  | Write or append to files                             |
| `file_list`   | List directory contents with glob filtering          |
| `code_search` | Search text across source files recursively          |
| `git_status`  | Show git status and recent log                       |
| `git_diff`    | Show uncommitted changes                             |
| `git_commit`  | Stage and commit changes                             |
| `git_branch`  | List, create, or checkout branches                   |
| `code_lint`   | Run ruff (Python) or eslint (JS/TS)                  |
| `test_runner` | Run pytest, jest, vitest, or go test                 |
| `web_fetch`   | Fetch documentation or API references from a URL     |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
