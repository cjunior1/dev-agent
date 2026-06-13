# Architecture — Cacau (dev-agent)

Cacau is a software development assistant built on top of LangGraph's ReAct agent loop. It exposes a CLI (`cacau`), an interactive REPL, and a webhook server. The Python package is `dev_agent` (under `src/`); the installed command is `cacau`.

---

## Package layout

```
src/dev_agent/
├── config.py              # Settings, LLMProfile, pydantic-settings + YAML
├── agent/
│   ├── harness.py         # AgentHarness — orchestrator, streaming, checkpointing
│   ├── graph.py           # LangGraph StateGraph (ReAct loop)
│   ├── state.py           # AgentState TypedDict
│   ├── providers.py       # LLM factory (Anthropic / OpenAI / Google / Groq / Ollama)
│   ├── selector.py        # Auto profile selection via classifier LLM
│   ├── prompts.py         # System prompt template
│   └── health.py          # Profile connectivity checks
├── cli/
│   ├── main.py            # Typer app — all CLI subcommands
│   └── repl.py            # Interactive REPL with Rich rendering
├── tools/
│   ├── registry.py        # Tool catalogue and group expansion
│   ├── shell.py           # shell tool (async subprocess, safety blocklist)
│   ├── filesystem.py      # file_read / file_write / file_list / code_search
│   ├── git_tools.py       # git_status / git_diff / git_commit / git_branch
│   ├── code_tools.py      # code_lint (ruff / eslint) / test_runner (pytest / jest / vitest / go)
│   └── web_tools.py       # web_fetch
└── webhooks/
    ├── server.py          # FastAPI app — /webhook/github, /webhook/gitlab, /webhook/run
    └── handlers.py        # GitHub / GitLab payload → WebhookTask / prompt
config/
└── settings.yaml          # Default profiles and runtime config
```

---

## Request flow

```
cacau run / chat
    │
    ▼
cli/main.py  ──────────────────────────────────────────────────────────┐
    │  builds AgentHarness(settings)                                   │
    │  calls harness.run(prompt, thread_id, workspace, profile, model) │
    ▼                                                                  │
AgentHarness.run()                                                     │
    │                                                                  │
    ├─ _resolve_profile()                                              │
    │       │                                                          │
    │       ├── explicit profile/model override → use directly        │
    │       └── "auto" mode → selector.select_profile()               │
    │               │  calls classifier LLM (llm_selector.profile)    │
    │               └─ returns profile name (fallback: first profile)  │
    │                                                                  │
    ├─ yields  { type: "profile_selected", ... }                       │
    │                                                                  │
    ├─ providers.build_llm(profile, tools)  → BaseChatModel + bind_tools
    │                                                                  │
    ├─ graph.build_graph(llm, settings, tools) → StateGraph           │
    │       agent node  ──→  ToolNode  ──→  agent node (loop)         │
    │       └── guarded by max_iterations                             │
    │                                                                  │
    ├─ graph.compile(MemorySaver, interrupt_before, interrupt_after)  │
    │                                                                  │
    └─ compiled.astream_events()   → yields typed events              │
            │                                                          │
            ├── on_chat_model_stream  →  { type: "token" }            │
            ├── on_tool_start         →  { type: "tool_call" }        │
            ├── on_tool_end           →  { type: "tool_result" }      │
            └── on_chain_end          →  { type: "done" }             │
                                                                       │
cli renders events ────────────────────────────────────────────────────┘
```

---

## Layers in detail

### 1. Configuration (`config.py`)

| Class | Purpose |
|---|---|
| `LLMProfile` | One named LLM endpoint — provider, model, api_key_env, base_url, temperature, streaming |
| `LLMSelectorConfig` | Which profile to use as the auto-selector classifier |
| `AgentConfig` | Default profile, max_iterations, recursion_limit |
| `HarnessConfig` | Checkpointing toggle, interrupt_before/after hook lists, debug_mode |
| `WebhookConfig` | Enabled flag, host, port, HMAC secret |
| `Settings` | Root object, env prefix `CACAU_`, nested delimiter `__` |

Config is loaded once from `config/settings.yaml` and cached via `get_settings()`. Tests call `reset_settings()` to bust the cache. Environment variables override YAML values (e.g. `CACAU_AGENT__PROFILE=fast`). The `.env` file is loaded automatically.

**Profiles** — each profile independently selects provider and model. The `api_key_env` field names the environment variable that holds the key; Ollama profiles omit it and use `base_url` instead.

### 2. Provider factory (`agent/providers.py`)

`build_llm(profile, tools) → BaseChatModel` performs three steps:

1. Resolves the API key from the named environment variable (`_resolve_api_key`).
2. Lazy-imports the correct LangChain integration class (prevents import errors when optional packages are absent).
3. Instantiates the model and calls `.bind_tools(tools)`.

Supported providers: `anthropic`, `openai`, `google`, `groq`, `ollama`.
A `ConfigError` is raised for missing environment variables or missing packages.

### 3. Auto-selector (`agent/selector.py`)

In `auto` mode `AgentHarness._resolve_profile()` calls `select_profile(prompt, profiles, classifier_llm)`.

The classifier LLM receives a structured prompt listing all profiles and their descriptions, then returns the name of the best match. On parse failure or an unknown name it falls back to the first profile in the dict. The classifier itself uses a lightweight, fast profile (default: `fast`).

### 4. Agent graph (`agent/graph.py`)

`build_graph(llm, settings, tools)` returns an uncompiled `StateGraph[AgentState]` with two nodes:

```
 ┌────────────────────────────────────────────────┐
 │                   StateGraph                   │
 │                                                │
 │   [START] ──→ [agent] ──→ [tools] ──┐         │
 │                  ↑                  │         │
 │                  └──────────────────┘         │
 │                  │                            │
 │            max_iterations?                    │
 │                  └──→ [END]                   │
 └────────────────────────────────────────────────┘
```

- **agent node** — prepends the system prompt to the message list, invokes the LLM, increments `tool_calls_count`.
- **should_continue** — delegates to `tools_condition` unless `tool_calls_count >= max_iterations`, in which case it forces `END`.
- **ToolNode** — executes whichever tools the LLM requested and appends results as `ToolMessage` objects.

The harness compiles the graph with a `MemorySaver` checkpointer (thread-scoped, in-memory), providing conversation continuity across calls on the same `thread_id`. Interrupt hooks (`interrupt_before` / `interrupt_after`) enable human-in-the-loop pauses; `harness.resume()` continues from the saved checkpoint.

### 5. Agent state (`agent/state.py`)

```python
class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]  # LangGraph reducer
    workspace: str
    tool_calls_count: int
    interrupted: bool
```

`add_messages` is the LangGraph reducer that merges message lists rather than overwriting them.

### 6. AgentHarness (`agent/harness.py`)

The harness is the single integration point between the CLI / webhook server and the LangGraph graph. It:

- Holds one `MemorySaver` instance (shared across threads/conversations).
- Builds the tool list once at `__init__` (`build_toolset(None)` → all tools).
- Builds the LLM and graph on every `run()` call — no stale state between invocations.
- Streams typed event dicts: `profile_selected`, `token`, `tool_call`, `tool_result`, `done`.
- Supports a `model` override that patches the resolved profile via `model_copy` without touching stored config.

### 7. Tools (`tools/`)

All tools are standard LangChain `@tool`-decorated functions, registered in `registry.py`.

| Tool | Module | Notes |
|---|---|---|
| `shell` | `shell.py` | Async subprocess, 60 s timeout, 8 kB output cap, hardcoded safety blocklist |
| `file_read` | `filesystem.py` | Line-numbered output, 12 kB cap |
| `file_write` | `filesystem.py` | Creates parent dirs, supports append |
| `file_list` | `filesystem.py` | Glob + recursive, 200 entry cap |
| `code_search` | `filesystem.py` | Case-insensitive grep across extensions, 100 match cap |
| `git_status` | `git_tools.py` | `git status --short` + `git log --oneline -10` |
| `git_diff` | `git_tools.py` | Configurable ref, 8 kB cap |
| `git_commit` | `git_tools.py` | Optional `add_all` (git add -A) |
| `git_branch` | `git_tools.py` | List / create / checkout |
| `code_lint` | `code_tools.py` | ruff or eslint, 120 s timeout |
| `test_runner` | `code_tools.py` | pytest / jest / vitest / go test |
| `web_fetch` | `web_tools.py` | HTTP GET for documentation |

`build_toolset(enabled)` expands group aliases (`"git"` → all `git_*` tools) and deduplicates by tool name.

### 8. CLI (`cli/main.py`, `cli/repl.py`)

Built with **Typer** + **Rich**. Command hierarchy:

```
cacau
├── run       <prompt>          # single-shot with streaming render
├── chat                        # interactive REPL (repl.py)
├── serve                       # webhook server (uvicorn)
├── tools                       # list available tools
└── config
    ├── show                    # dump active config as YAML
    ├── check                   # health-check all profiles concurrently
    ├── profile
    │   ├── list
    │   ├── add    [--provider --model ...]  # interactive or flag-driven
    │   ├── edit   <name> [flags]
    │   ├── remove <name>
    │   └── use    <name>        # sets agent.profile in settings.yaml
    └── key
        ├── set    <KEY> <value>  # writes/updates .env
        └── list                 # masked display
```

Config writes use a raw YAML round-trip (`_read_yaml_raw` / `_write_yaml_raw`) rather than serialising Pydantic models, which preserves unrecognised keys. API key writes use `_write_env_key` which updates the `.env` file in-place. After any write, `reset_settings()` clears the singleton cache.

The **REPL** (`repl.py`) maintains a single `thread_id` per session, persists input history to `~/.cacau_history`, and supports slash commands (`/help`, `/tools`, `/history`, `/thread`, `/threads`, `/new`, `/clear`, `/exit`).

### 9. Webhook server (`webhooks/`)

`cacau serve` starts a **FastAPI** / **uvicorn** server. Three endpoints:

| Endpoint | Auth | Description |
|---|---|---|
| `POST /webhook/github` | HMAC-SHA256 (`X-Hub-Signature-256`) | GitHub PR / push / issue events |
| `POST /webhook/gitlab` | Token header (`X-Gitlab-Token`) | GitLab MR / push / issue events |
| `POST /webhook/run` | None (optional secret) | Generic `{prompt, workspace?, thread_id?}` |
| `GET /health` | None | Liveness probe |

`handlers.py` maps each payload shape to a `WebhookTask` dataclass carrying a `prompt`, `workspace`, `thread_id`, and metadata. The agent is dispatched as a **FastAPI background task** so the webhook endpoint returns `202 Accepted` immediately. Each webhook call creates a new `thread_id` (no cross-event conversation continuity by default).

---

## Configuration reference

```yaml
profiles:
  <name>:
    provider: anthropic | openai | google | groq | ollama
    model: <model-id>
    api_key_env: <ENV_VAR_NAME>     # omit for Ollama
    base_url: <url>                 # Ollama / proxy
    temperature: 0.1
    streaming: true
    description: "..."

agent:
  profile: auto | <profile-name>
  max_iterations: 25
  recursion_limit: 50
  streaming: true

llm_selector:
  profile: fast                     # classifier profile for auto mode

harness:
  checkpointing: true
  interrupt_before: []
  interrupt_after: []
  debug_mode: false

webhooks:
  enabled: false
  host: "0.0.0.0"
  port: 8080
  secret: ""
```

Environment overrides use `CACAU_` prefix with `__` as the nested delimiter:

```
CACAU_AGENT__PROFILE=balanced
CACAU_AGENT__MAX_ITERATIONS=50
CACAU_HARNESS__DEBUG_MODE=true
```

---

## Key design decisions

**Lazy provider imports** — each `_import_<provider>()` function in `providers.py` defers the import so a missing optional package only raises `ConfigError` when that provider is actually selected, not at module load time.

**Graph rebuilt per call** — `AgentHarness.run()` builds a new LLM and graph on every invocation. The `MemorySaver` checkpointer is shared, so thread history is preserved across calls even though graph objects are not reused.

**Streaming via `astream_events` v2** — the harness filters `on_chat_model_stream`, `on_tool_start`, `on_tool_end`, and `on_chain_end` events and re-emits them as simple dicts. Consumers (CLI, REPL, webhook dispatcher) do not need to understand LangGraph internals.

**Settings singleton** — `get_settings()` caches the `Settings` instance globally. Tests call `reset_settings()` after patching env vars to avoid cache bleed. All config writes in the CLI also call `reset_settings()`.

**Tool output truncation** — each tool caps its output at a fixed limit (shell: 8 kB, file_read: 12 kB, git_diff: 8 kB, tool_result in harness: 2 kB) to prevent context window saturation.
