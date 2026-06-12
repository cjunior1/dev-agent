# Multi-LLM Profiles & Auto-Selection — Design Spec

**Date:** 2026-06-12
**Status:** Approved

## Problem

The agent currently hardcodes `ChatAnthropic` in `agent/graph.py` and accepts only a single `model` string in config. There is no way to switch providers, configure multiple models, or let the agent choose the best LLM for a given task.

## Goal

- Support any LLM provider through named profiles defined in `config/settings.yaml`
- Add an `auto` mode where a lightweight LLM classifier reads profile descriptions and selects the most appropriate profile for each prompt
- Validate profiles at startup via `dev-agent config check`
- Show the selected profile in the REPL when auto mode is active

---

## Supported Providers

| Provider  | LangChain class               | Auth            |
|-----------|-------------------------------|-----------------|
| anthropic | `ChatAnthropic`               | `ANTHROPIC_API_KEY` |
| openai    | `ChatOpenAI`                  | `OPENAI_API_KEY` |
| google    | `ChatGoogleGenerativeAI`      | `GOOGLE_API_KEY` |
| ollama    | `ChatOllama`                  | none (local)    |
| groq      | `ChatGroq`                    | `GROQ_API_KEY`  |

Provider packages are optional extras in `pyproject.toml`; missing packages produce a clear install hint.

---

## Config Schema

`config/settings.yaml` replaces the flat `agent.model` field with a `profiles` section and `agent.profile`:

```yaml
profiles:
  powerful:
    provider: anthropic
    model: claude-opus-4-8
    api_key_env: ANTHROPIC_API_KEY          # env var name (not the value)
    temperature: 0.1
    streaming: true
    description: >
      Best for complex reasoning, architecture decisions, and reviewing large codebases.

  fast:
    provider: groq
    model: llama-3.3-70b-versatile
    api_key_env: GROQ_API_KEY
    temperature: 0.2
    streaming: true
    description: >
      Fast and cheap. Ideal for simple tasks: formatting, explaining short functions,
      quick Q&A, and anything that doesn't require deep reasoning.

  balanced:
    provider: openai
    model: gpt-4o-mini
    api_key_env: OPENAI_API_KEY
    temperature: 0.1
    streaming: true
    description: >
      Good balance of cost and quality. Suitable for debugging, writing unit tests,
      and medium-complexity refactors.

  local:
    provider: ollama
    model: qwen2.5-coder:7b
    base_url: http://localhost:11434
    description: >
      Local model, no cost, no data leaves the machine. Use for sensitive codebases
      or offline environments.

agent:
  profile: auto          # name of a profile, or "auto"
  max_iterations: 25
  recursion_limit: 50
  streaming: true

llm_selector:
  profile: fast          # profile to use as the classifier in auto mode
```

### Rules
- `agent.profile` can be any profile name or the literal `"auto"`.
- `api_key_env` stores the env var *name*, never the key value.
- `base_url` is only valid for `ollama`.
- `temperature` and `streaming` at the profile level override agent-level defaults.

---

## New Files

### `agent/providers.py`
Factory that instantiates the correct `BaseChatModel` for a given `LLMProfile`. Each provider is imported lazily so missing optional packages fail only when that provider is actually used.

**Signature:**
```python
def build_llm(profile: LLMProfile, tools: list[BaseTool]) -> BaseChatModel
```

### `agent/selector.py`
Selects the best profile for a given prompt using the configured classifier LLM.

**Flow:**
1. Build the classifier LLM from `settings.llm_selector.profile`
2. Build a structured prompt: classifier role + list of `"<name>: <description>"` + user prompt
3. Call the classifier, parse the response to extract a profile name
4. Validate: if name not in profiles, fall back to the first available profile
5. Return the profile name and log the selection reason

**Classifier prompt template:**
```
You are an LLM router. Given the user's task and the available LLM profiles below,
reply with ONLY the name of the most suitable profile — nothing else.

Profiles:
{profiles}

User task: {prompt}
```

### `agent/health.py`
Health check logic for `dev-agent config check`:
- For each configured profile, build the LLM and call it with `"What is the capital of France?"`
- Record: latency, response snippet (or error), and whether the API key env var is set
- Return a list of `ProfileStatus(name, ok, latency_ms, error)` objects

---

## Modified Files

### `config.py`
- Add `LLMProfile` Pydantic model with fields: `provider`, `model`, `api_key_env`, `base_url`, `temperature`, `streaming`, `description`
- Add `LLMSelectorConfig` with `profile: str`
- Add `profiles: dict[str, LLMProfile]` and `llm_selector: LLMSelectorConfig` to `Settings`
- Remove `AgentConfig.model` (replaced by `profile`)

### `agent/graph.py`
- Remove `ChatAnthropic` import and hardcoded instantiation
- Accept a `BaseChatModel` (already bound to tools) instead of building it internally
- Signature: `build_graph(llm: BaseChatModel, settings: Settings)`

### `agent/harness.py`
- On `run()`: if `settings.agent.profile == "auto"`, call `selector.select(prompt, profiles)` first
- Build the LLM via `providers.build_llm(selected_profile, tools)`
- Pass selected profile name in yielded events: `{"type": "profile_selected", "payload": name}`

### `cli/main.py`
- Add `dev-agent config check` subcommand that calls `health.check_all()` and renders results as a Rich table

### `cli/repl.py`
- Handle `profile_selected` event: render `[auto → <name> · <model>]` badge before the first token

### `pyproject.toml`
- Add optional dependency groups per provider:
  ```toml
  [project.optional-dependencies]
  anthropic = ["langchain-anthropic>=0.3.0"]
  openai    = ["langchain-openai>=0.2.0"]
  google    = ["langchain-google-genai>=2.0.0"]
  groq      = ["langchain-groq>=0.2.0"]
  all-llms  = ["langchain-anthropic", "langchain-openai", "langchain-google-genai", "langchain-groq"]
  ```
  Ollama uses `langchain-community` (already a dependency).

---

## Data Flow — Auto Mode

```
user prompt
    │
    ▼
selector.select(prompt, profiles)
    │  calls classifier LLM (fast profile)
    │  returns profile name, e.g. "powerful"
    ▼
providers.build_llm(profiles["powerful"], tools)
    │  returns ChatAnthropic bound to tools
    ▼
build_graph(llm, settings)
    │
    ▼
harness streams events → repl renders badge + tokens
```

---

## CLI Output

### `dev-agent config check`
```
Checking LLM profiles...
  ✓ powerful   anthropic / claude-opus-4-8        142ms   "The capital of France is Paris."
  ✓ fast       groq / llama-3.3-70b-versatile      89ms   "Paris."
  ✗ balanced   openai / gpt-4o-mini               —       OPENAI_API_KEY not set
  ✓ local      ollama / qwen2.5-coder:7b           310ms  "Paris."

3/4 profiles healthy.
```

### REPL badge (auto mode)
```
you> refactor the payments module and add tests

[auto → powerful · claude-opus-4-8]

⚙ file_read  (path=src/payments/processor.py)
  → ...
```

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| API key env var not set | `providers.build_llm` raises `ConfigError` with install hint |
| Provider package not installed | Same, with `pip install dev-agent[<provider>]` hint |
| Classifier returns unknown profile name | Falls back to first available healthy profile, logs warning |
| Classifier call fails | Falls back to first available healthy profile, logs error |
| All profiles unhealthy | Hard error at startup with actionable message |

---

## Verification

```bash
# Install with all provider extras
pip install -e ".[all-llms,dev]"

# Check profile health
dev-agent config check

# Use a specific profile
dev-agent run "explain this function" --profile fast

# Use auto mode (default)
dev-agent chat

# Run tests
pytest tests/ -v
```
