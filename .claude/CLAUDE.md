# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Reference

See `ARCHITECTURE.md` in the project root for full architecture details. Read it when analyzing the codebase, debugging, or before making structural changes.

## Commands

```bash
# Install (venv is at .venv/)
pip install -e ".[all-llms,dev]"

# Run all tests
.venv/bin/pytest tests/ -v

# Run a single test file
.venv/bin/pytest tests/test_tools.py -v

# Run a single test
.venv/bin/pytest tests/test_selector.py::test_select_returns_valid_profile -v

# Lint
.venv/bin/ruff check src/

# CLI (installed as 'cacau')
.venv/bin/cacau --help
.venv/bin/cacau config show
.venv/bin/cacau config check
```

## Architecture

The Python package is `dev_agent` (under `src/`) but the CLI command is `cacau`.

**Request flow:**

```
cacau run/chat  →  cli/main.py  →  AgentHarness.run()
                                        │
                          _resolve_profile()  ←  selector.py (auto mode)
                                        │
                          providers.build_llm()  ←  config.LLMProfile
                                        │
                          build_graph(llm, settings)
                                        │
                          compiled.astream_events()  →  yields typed events
                                        │
                          cli renders: token / tool_call / tool_result / done
```

**Key layers:**

- **`config.py`** — `Settings` (pydantic-settings, env prefix `CACAU_`), `LLMProfile`, `LLMSelectorConfig`. Loaded once via `get_settings()` (cached); `reset_settings()` clears the cache. Config file is `config/settings.yaml`; env file is `.env`.

- **`agent/providers.py`** — `build_llm(profile, tools)` factory. Provider packages are optional extras; imports are lazy so a missing package only fails when that provider is actually used. Raises `ConfigError` for missing keys or packages.

- **`agent/selector.py`** — `select_profile(prompt, profiles, classifier_llm)` async. Calls the classifier LLM and returns a profile name; falls back to the first profile on any error.

- **`agent/harness.py`** — `AgentHarness`. Builds LLM and graph on every `run()` call (not at `__init__`). Yields a `profile_selected` event before the first token. Accepts `profile` and `model` overrides per call; `model` patches the resolved profile via `model_copy(update={"model": ...})`.

- **`agent/graph.py`** — `build_graph(llm, settings, tools)` returns an uncompiled `StateGraph`. The harness compiles it with `MemorySaver` checkpointer and interrupt hooks. The executable `tools` list is passed in explicitly to populate the `ToolNode` — the bound LLM does not expose the original tool objects (`bind_tools` returns a `RunnableBinding` whose tools live as schema dicts in `.kwargs`, not as a `.tools` attribute).

- **`agent/health.py`** — `check_all(profiles)` runs `check_profile` concurrently via `asyncio.gather`. Used by `cacau config check`.

- **`cli/main.py`** — Typer app (`cacau`). Sub-apps: `config_app → profile_app`, `key_app`. Config writes go through `_read_yaml_raw` / `_write_yaml_raw` (raw dict round-trip through PyYAML — comments are not preserved). API key writes go through `_write_env_key` which updates `.env` in-place.

- **`tools/registry.py`** — `build_toolset(enabled)` returns the tool list; `None` means all tools.

- **`tools/filesystem.py`** — file tools. **Writes are confined to a workspace root; reads/search are not.** Relative paths in all four tools (`file_read`, `file_write`, `file_list`, `code_search`) resolve against the workspace root via `_resolve`; absolute paths are used as-is. `file_write` additionally refuses any target outside the workspace (incl. `../` traversal and symlink escape, since paths are `.resolve()`d). The workspace root lives in a `ContextVar` (`set_workspace_root`), so concurrent runs (e.g. the webhook server) stay isolated. The harness sets it per `run()` and restores it per `resume()` (tracked by `thread_id` in `AgentHarness._workspaces`). Default root is the process CWD.

## Testing conventions

- `pytest-asyncio` in **strict mode** — every async test requires `@pytest.mark.asyncio`.
- Use `asyncio.run(coro)`, never `asyncio.get_event_loop().run_until_complete()`.
- Provider tests mock lazy-import helpers (e.g. `patch("dev_agent.agent.providers._import_anthropic")`).
- After modifying `Settings` in a test, call `reset_settings()` so the cache doesn't bleed into other tests.
- Tests exercising `file_write` outside the CWD must call `set_workspace_root(...)` first, or the write is refused as outside the workspace.

## LLM profile config

Profiles live in `config/settings.yaml`. `agent.profile` can be a profile name or `"auto"`. In auto mode, `llm_selector.profile` names the classifier profile. The env prefix for overrides is `CACAU_` with `__` as delimiter (e.g. `CACAU_AGENT__PROFILE=fast`).

Ollama profiles omit `api_key_env` and use `base_url`. All other providers require `api_key_env` to name the env var holding the key.
