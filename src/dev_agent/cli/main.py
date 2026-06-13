"""Dev Agent CLI — entry point with Typer subcommands."""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

app = typer.Typer(
    name="cacau",
    help="Cacau — software development assistant powered by LangGraph Deep Agent.",
    no_args_is_help=True,
)
console = Console()
config_app = typer.Typer(help="Configuration commands.")
profile_app = typer.Typer(help="Manage LLM profiles.")
key_app = typer.Typer(help="Manage API keys in .env.")
app.add_typer(config_app, name="config")
config_app.add_typer(profile_app, name="profile")
config_app.add_typer(key_app, name="key")

# Paths
_SETTINGS_PATH = Path(__file__).parent.parent.parent.parent / "config" / "settings.yaml"
_ENV_PATH = Path(__file__).parent.parent.parent.parent / ".env"

_PROVIDER_DEFAULTS: dict[str, dict] = {
    "anthropic": {"model": "claude-sonnet-4-6",       "api_key_env": "ANTHROPIC_API_KEY"},
    "openai":    {"model": "gpt-4o-mini",             "api_key_env": "OPENAI_API_KEY"},
    "google":    {"model": "gemini-1.5-flash",        "api_key_env": "GOOGLE_API_KEY"},
    "groq":      {"model": "llama-3.3-70b-versatile", "api_key_env": "GROQ_API_KEY"},
    "ollama":    {"model": "qwen2.5-coder:7b",        "api_key_env": None},
}


# ── File I/O helpers ──────────────────────────────────────────────────────────

def _read_yaml_raw(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _write_yaml_raw(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _write_env_key(key: str, value: str, env_path: Path) -> None:
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}=") or line.startswith(f"{key} ="):
            lines[i] = f"{key}={value}"
            updated = True
            break
    if not updated:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n")


def _read_env(env_path: Path) -> dict[str, str]:
    result = {}
    if not env_path.exists():
        return result
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        result[k.strip()] = v.strip()
    return result


def _mask(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return value[:4] + "***" + value[-4:]


# ── Main agent helpers ────────────────────────────────────────────────────────

def _get_harness(workspace: str):
    from dev_agent.agent.harness import AgentHarness
    from dev_agent.config import get_settings
    settings = get_settings()
    return AgentHarness(settings), workspace


# ── run ───────────────────────────────────────────────────────────────────────

@app.command("run")
def run_cmd(
    prompt: str = typer.Argument(..., help="Prompt to send to the agent."),
    workspace: str = typer.Option(".", "--workspace", "-w", help="Working directory for tools."),
    thread_id: Optional[str] = typer.Option(None, "--thread", "-t", help="Thread ID (resumes conversation)."),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="LLM profile name (overrides config)."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name (overrides the profile's model)."),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON events instead of rendered output."),
):
    """Run the agent with a single prompt and stream the response."""

    async def _run():
        harness, ws = _get_harness(workspace)
        tid = thread_id or harness.new_thread()

        async for event in harness.run(prompt, thread_id=tid, workspace=ws, profile=profile, model=model):
            if json_output:
                print(json.dumps(event), flush=True)
                continue

            etype, payload = event["type"], event["payload"]
            if etype == "profile_selected":
                console.print(
                    f"\n[dim][auto → [cyan]{payload['name']}[/] · {payload['model']}][/dim]\n"
                    if (profile is None and harness.settings.agent.profile == "auto") else ""
                )
            elif etype == "token":
                print(payload, end="", flush=True)
            elif etype == "tool_call":
                console.print(f"\n[cyan]⚙ {payload['tool']}[/]", end=" ")
                args_str = ", ".join(f"{k}={v!r}" for k, v in payload.get("input", {}).items())
                console.print(f"[dim]({args_str})[/dim]")
            elif etype == "tool_result":
                out = payload.get("output", "")[:400]
                console.print(f"[dim]  → {out}[/dim]")
            elif etype == "done":
                print()
                console.print(f"\n[dim]thread: {tid}[/dim]")

    asyncio.run(_run())


# ── chat ──────────────────────────────────────────────────────────────────────

@app.command("chat")
def chat_cmd(
    workspace: str = typer.Option(".", "--workspace", "-w", help="Working directory for tools."),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="LLM profile name (overrides config)."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name (overrides the profile's model)."),
):
    """Start an interactive chat REPL with the agent."""
    from dev_agent.cli.repl import run_repl

    async def _chat():
        harness, ws = _get_harness(workspace)
        await run_repl(harness, workspace=ws, default_profile=profile, default_model=model)

    asyncio.run(_chat())


# ── serve ─────────────────────────────────────────────────────────────────────

@app.command("serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host."),
    port: int = typer.Option(8080, "--port", "-p", help="Bind port."),
    workspace: str = typer.Option(".", "--workspace", "-w", help="Default workspace."),
    secret: str = typer.Option("", "--secret", help="HMAC secret for webhook verification."),
):
    """Start the webhook server for GitHub/GitLab event processing."""
    import uvicorn
    from dev_agent.webhooks.server import create_app
    from dev_agent.config import get_settings

    settings = get_settings()
    if secret:
        settings.webhooks.secret = secret

    harness, ws = _get_harness(workspace)
    web_app = create_app(harness, default_workspace=ws, settings=settings)

    console.print(Panel(
        f"[bold green]Cacau Webhook Server[/bold green]\n"
        f"Listening on [cyan]http://{host}:{port}[/cyan]\n"
        f"Workspace: [dim]{ws}[/dim]",
        border_style="green",
    ))
    uvicorn.run(web_app, host=host, port=port, log_level="warning")


# ── tools ─────────────────────────────────────────────────────────────────────

@app.command("tools")
def tools_cmd():
    """List all available agent tools."""
    from dev_agent.tools.registry import list_tools

    table = Table(title="Available Tools", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description")

    for name, desc in list_tools().items():
        table.add_row(name, desc)

    console.print(table)


# ── config show / check ───────────────────────────────────────────────────────

@config_app.command("show")
def config_show_cmd():
    """Show the active configuration."""
    from dev_agent.config import get_settings

    settings = get_settings()
    data = {
        "agent": settings.agent.model_dump(),
        "llm_selector": settings.llm_selector.model_dump(),
        "profiles": {name: p.model_dump() for name, p in settings.profiles.items()},
        "harness": settings.harness.model_dump(),
        "webhooks": settings.webhooks.model_dump(),
    }
    console.print(Syntax(yaml.dump(data, default_flow_style=False), "yaml", theme="monokai"))


@config_app.command("check")
def config_check_cmd():
    """Test connectivity for all configured LLM profiles."""
    from dev_agent.agent.health import check_all
    from dev_agent.config import get_settings

    settings = get_settings()
    console.print("\n[bold]Checking LLM profiles...[/bold]\n")

    statuses = asyncio.run(check_all(settings.profiles))

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("", width=3)
    table.add_column("Profile", style="cyan", no_wrap=True, min_width=12)
    table.add_column("Provider / Model", min_width=30)
    table.add_column("Latency", justify="right", min_width=8)
    table.add_column("Result / Error")

    ok_count = 0
    for s in statuses:
        icon = "[green]✓[/]" if s.ok else "[red]✗[/]"
        provider_model = f"{s.provider} / {s.model}"
        latency = f"{s.latency_ms:.0f}ms" if s.ok else "—"
        result = f'[dim]"{s.snippet}"[/dim]' if s.ok else f"[red]{s.error}[/red]"
        table.add_row(icon, s.name, provider_model, latency, result)
        if s.ok:
            ok_count += 1

    console.print(table)
    total = len(statuses)
    colour = "green" if ok_count == total else "yellow" if ok_count > 0 else "red"
    console.print(f"\n[{colour}]{ok_count}/{total} profiles healthy.[/]\n")


# ── config profile ────────────────────────────────────────────────────────────

@profile_app.command("list")
def profile_list_cmd():
    """List all configured LLM profiles."""
    from dev_agent.config import get_settings

    settings = get_settings()
    active = settings.agent.profile

    table = Table(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
    table.add_column("", width=3)
    table.add_column("Name", style="cyan", no_wrap=True, min_width=14)
    table.add_column("Provider", min_width=10)
    table.add_column("Model", min_width=28)
    table.add_column("API Key Env", min_width=20)

    for name, p in settings.profiles.items():
        icon = "[green]✓[/]" if name == active else "[dim]·[/dim]"
        api_key = p.api_key_env or "[dim]—[/dim]"
        table.add_row(icon, name, p.provider, p.model, api_key)

    console.print()
    console.print(table)
    console.print(f"\n[dim]Default profile: [cyan]{active}[/cyan][/dim]\n")


@profile_app.command("add")
def profile_add_cmd(
    name: Optional[str] = typer.Argument(None, help="Profile name (prompted if omitted)."),
    provider: Optional[str] = typer.Option(None, "--provider", help="Provider: anthropic, openai, google, groq, ollama."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name (e.g. claude-opus-4-8)."),
    api_key_env: Optional[str] = typer.Option(None, "--api-key-env", help="Env var holding the API key."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Base URL (for Ollama or proxies)."),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Profile description."),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="Sampling temperature (default 0.1)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Add or update an LLM profile. Omitted fields are prompted interactively."""
    from dev_agent.config import reset_settings

    providers = list(_PROVIDER_DEFAULTS.keys())
    interactive = not all([provider, model])

    if interactive:
        console.print("\n[bold cyan]Add LLM Profile[/bold cyan]\n")

    # Name
    if not name:
        name = typer.prompt("Profile name")
    name = name.strip()

    data = _read_yaml_raw(_SETTINGS_PATH)
    existing_profiles = data.get("profiles", {})

    if name in existing_profiles and not yes:
        overwrite = typer.confirm(f"Profile '{name}' already exists. Overwrite?", default=False)
        if not overwrite:
            raise typer.Abort()

    # Provider
    if not provider:
        console.print(f"Providers: {', '.join(providers)}")
        provider = typer.prompt("Provider", default="anthropic")
        while provider not in providers:
            console.print(f"[red]Unknown provider. Choose from: {', '.join(providers)}[/red]")
            provider = typer.prompt("Provider", default="anthropic")
    elif provider not in providers:
        console.print(f"[red]Unknown provider '{provider}'. Choose from: {', '.join(providers)}[/red]")
        raise typer.Exit(1)

    defaults = _PROVIDER_DEFAULTS[provider]

    # Model
    if not model:
        model = typer.prompt("Model", default=defaults["model"])

    # API key env var
    if provider == "ollama":
        api_key_env = None
    elif api_key_env is None:
        raw = typer.prompt("API key env var", default=defaults["api_key_env"] or "")
        api_key_env = raw.strip() or None

    # Base URL
    if provider == "ollama" and base_url is None:
        base_url = typer.prompt("Base URL", default="http://localhost:11434")
    elif base_url is None and interactive:
        if typer.confirm("Set a custom base URL? (for proxies or local endpoints)", default=False):
            base_url = typer.prompt("Base URL")

    # Description
    if description is None and interactive:
        description = typer.prompt("Description (optional)", default="") or None

    # Temperature
    if temperature is None and interactive:
        raw_temp = typer.prompt("Temperature", default="0.1")
        try:
            temperature = float(raw_temp)
        except ValueError:
            temperature = 0.1
    temperature = temperature if temperature is not None else 0.1

    # Build profile dict
    profile_dict: dict = {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "streaming": True,
    }
    if api_key_env:
        profile_dict["api_key_env"] = api_key_env
    if base_url:
        profile_dict["base_url"] = base_url
    if description:
        profile_dict["description"] = description

    if not yes:
        console.print()
        console.print(Panel(
            Syntax(yaml.dump({name: profile_dict}, default_flow_style=False), "yaml", theme="monokai"),
            title="Profile preview",
            border_style="cyan",
        ))
        if not typer.confirm("Save this profile?", default=True):
            raise typer.Abort()

    if "profiles" not in data:
        data["profiles"] = {}
    data["profiles"][name] = profile_dict
    _write_yaml_raw(data, _SETTINGS_PATH)
    reset_settings()

    console.print(f"\n[green]✓[/green] Profile '[cyan]{name}[/cyan]' saved to {_SETTINGS_PATH.name}\n")
    if api_key_env:
        console.print(f"[dim]Don't forget to set [cyan]{api_key_env}[/cyan] — run:[/dim]")
        console.print(f"  [bold]cacau config key set {api_key_env} <your-key>[/bold]\n")


@profile_app.command("edit")
def profile_edit_cmd(
    name: str = typer.Argument(..., help="Profile name to edit."),
    provider: Optional[str] = typer.Option(None, "--provider", help="Provider: anthropic, openai, google, groq, ollama."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name."),
    api_key_env: Optional[str] = typer.Option(None, "--api-key-env", help="Env var holding the API key."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Base URL (for Ollama or proxies). Pass empty string to remove."),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Profile description."),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="Sampling temperature."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Edit an existing LLM profile. Omitted flags are prompted with current values as default."""
    from dev_agent.config import reset_settings

    data = _read_yaml_raw(_SETTINGS_PATH)
    profiles = data.get("profiles", {})

    if name not in profiles:
        console.print(f"[red]Profile '{name}' not found.[/red]")
        console.print(f"[dim]Available: {', '.join(profiles.keys())}[/dim]")
        raise typer.Exit(1)

    current = profiles[name]
    providers = list(_PROVIDER_DEFAULTS.keys())
    non_interactive = any([provider, model, api_key_env, base_url, description, temperature is not None])

    if not non_interactive:
        console.print(f"\n[bold cyan]Edit profile '[/bold cyan][cyan]{name}[/cyan][bold cyan]'[/bold cyan] — press Enter to keep current value\n")

    # Provider
    if provider is None:
        provider = typer.prompt("Provider", default=current.get("provider", "anthropic"))
        while provider not in providers:
            console.print(f"[red]Unknown provider. Choose from: {', '.join(providers)}[/red]")
            provider = typer.prompt("Provider", default=current.get("provider", "anthropic"))

    # Model
    if model is None:
        model = typer.prompt("Model", default=current.get("model", ""))

    # API key env var
    if provider == "ollama":
        api_key_env = None
    elif api_key_env is None:
        raw = typer.prompt("API key env var", default=current.get("api_key_env") or "")
        api_key_env = raw.strip() or None

    # Base URL
    if base_url is None:
        if provider == "ollama":
            base_url = typer.prompt("Base URL", default=current.get("base_url") or "http://localhost:11434")
        elif current.get("base_url") or not non_interactive:
            raw = typer.prompt("Base URL (leave blank to remove)", default=current.get("base_url") or "")
            base_url = raw.strip() or None
    else:
        base_url = base_url.strip() or None

    # Description
    if description is None:
        description = typer.prompt("Description", default=current.get("description") or "")
    description = description.strip() or None

    # Temperature
    if temperature is None:
        raw_temp = typer.prompt("Temperature", default=str(current.get("temperature", 0.1)))
        try:
            temperature = float(raw_temp)
        except ValueError:
            temperature = current.get("temperature", 0.1)

    updated: dict = {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "streaming": current.get("streaming", True),
    }
    if api_key_env:
        updated["api_key_env"] = api_key_env
    if base_url:
        updated["base_url"] = base_url
    if description:
        updated["description"] = description

    if not yes:
        console.print()
        console.print(Panel(
            Syntax(yaml.dump({name: updated}, default_flow_style=False), "yaml", theme="monokai"),
            title="Updated profile",
            border_style="cyan",
        ))
        if not typer.confirm("Save changes?", default=True):
            raise typer.Abort()

    data["profiles"][name] = updated
    _write_yaml_raw(data, _SETTINGS_PATH)
    reset_settings()
    console.print(f"\n[green]✓[/green] Profile '[cyan]{name}[/cyan]' updated.\n")


@profile_app.command("remove")
def profile_remove_cmd(
    name: str = typer.Argument(..., help="Profile name to remove."),
):
    """Remove an LLM profile."""
    from dev_agent.config import reset_settings

    data = _read_yaml_raw(_SETTINGS_PATH)
    profiles = data.get("profiles", {})

    if name not in profiles:
        console.print(f"[red]Profile '{name}' not found.[/red]")
        raise typer.Exit(1)

    active = (data.get("agent") or {}).get("profile", "auto")
    if name == active:
        console.print(f"[yellow]Warning:[/yellow] '{name}' is the current default profile (agent.profile = {active}).")

    confirm = typer.confirm(f"Remove profile '{name}'?", default=False)
    if not confirm:
        raise typer.Abort()

    del data["profiles"][name]
    _write_yaml_raw(data, _SETTINGS_PATH)
    reset_settings()
    console.print(f"[green]✓[/green] Profile '[cyan]{name}[/cyan]' removed.\n")


@profile_app.command("use")
def profile_use_cmd(
    name: str = typer.Argument(..., help="Profile name to set as default."),
):
    """Set the default LLM profile (writes agent.profile in settings.yaml)."""
    from dev_agent.config import reset_settings

    data = _read_yaml_raw(_SETTINGS_PATH)
    profiles = data.get("profiles", {})

    valid = list(profiles.keys()) + ["auto"]
    if name not in valid:
        console.print(f"[red]Profile '{name}' not found.[/red] Available: {', '.join(profiles.keys())}, auto")
        raise typer.Exit(1)

    if "agent" not in data:
        data["agent"] = {}
    data["agent"]["profile"] = name
    _write_yaml_raw(data, _SETTINGS_PATH)
    reset_settings()
    console.print(f"[green]✓[/green] Default profile set to '[cyan]{name}[/cyan]'.\n")


# ── config key ────────────────────────────────────────────────────────────────

@key_app.command("set")
def key_set_cmd(
    key: str = typer.Argument(..., help="Environment variable name (e.g. ANTHROPIC_API_KEY)."),
    value: str = typer.Argument(..., help="API key value."),
):
    """Set an API key in the .env file."""
    _write_env_key(key, value, _ENV_PATH)
    console.print(f"[green]✓[/green] [cyan]{key}[/cyan] written to [dim]{_ENV_PATH.name}[/dim]")

    known_keys = {v["api_key_env"] for v in _PROVIDER_DEFAULTS.values() if v["api_key_env"]}
    if key in known_keys:
        console.print("[dim]Run [bold]dev-agent config check[/bold] to verify connectivity.[/dim]\n")


@key_app.command("list")
def key_list_cmd():
    """List API keys set in the .env file (values are masked)."""
    env = _read_env(_ENV_PATH)

    if not env:
        console.print(f"\n[dim]No .env file found at {_ENV_PATH}[/dim]")
        console.print("[dim]Run [bold]dev-agent config key set KEY value[/bold] to create one.[/dim]\n")
        return

    table = Table(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
    table.add_column("Key", style="cyan", no_wrap=True, min_width=28)
    table.add_column("Value")

    for k, v in env.items():
        table.add_row(k, _mask(v))

    console.print()
    console.print(table)
    console.print(f"\n[dim]Source: {_ENV_PATH}[/dim]\n")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    app()


if __name__ == "__main__":
    main()
