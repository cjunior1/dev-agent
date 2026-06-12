"""Dev Agent CLI — entry point with Typer subcommands."""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

app = typer.Typer(
    name="dev-agent",
    help="Software development assistant powered by LangGraph Deep Agent.",
    no_args_is_help=True,
)
console = Console()


def _get_harness(workspace: str):
    from dev_agent.agent.harness import AgentHarness
    from dev_agent.config import get_settings

    settings = get_settings()
    if not settings.anthropic_api_key:
        console.print("[red]Error:[/] ANTHROPIC_API_KEY is not set. Add it to .env or export it.")
        raise typer.Exit(1)
    return AgentHarness(settings), workspace


@app.command("run")
def run_cmd(
    prompt: str = typer.Argument(..., help="Prompt to send to the agent."),
    workspace: str = typer.Option(".", "--workspace", "-w", help="Working directory for tools."),
    thread_id: Optional[str] = typer.Option(None, "--thread", "-t", help="Thread ID (resumes conversation)."),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON events instead of rendered output."),
):
    """Run the agent with a single prompt and stream the response."""

    async def _run():
        harness, ws = _get_harness(workspace)
        tid = thread_id or harness.new_thread()

        async for event in harness.run(prompt, thread_id=tid, workspace=ws):
            if json_output:
                print(json.dumps(event), flush=True)
                continue

            etype, payload = event["type"], event["payload"]
            if etype == "token":
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
                if not any(True for e in [event] if e["type"] == "token"):
                    console.print(payload)
                console.print(f"\n[dim]thread: {tid}[/dim]")

    asyncio.run(_run())


@app.command("chat")
def chat_cmd(
    workspace: str = typer.Option(".", "--workspace", "-w", help="Working directory for tools."),
):
    """Start an interactive chat REPL with the agent."""
    from dev_agent.cli.repl import run_repl

    async def _chat():
        harness, ws = _get_harness(workspace)
        await run_repl(harness, workspace=ws)

    asyncio.run(_chat())


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host."),
    port: int = typer.Option(8080, "--port", "-p", help="Bind port."),
    workspace: str = typer.Option(".", "--workspace", "-w", help="Default workspace for webhook tasks."),
    secret: str = typer.Option("", "--secret", help="HMAC secret for webhook signature verification."),
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
        f"[bold green]Dev Agent Webhook Server[/bold green]\n"
        f"Listening on [cyan]http://{host}:{port}[/cyan]\n"
        f"Workspace: [dim]{ws}[/dim]\n"
        f"Endpoints: /health  /webhook/github  /webhook/gitlab",
        border_style="green",
    ))
    uvicorn.run(web_app, host=host, port=port, log_level="warning")


@app.command("config")
def config_cmd(
    action: str = typer.Argument("show", help="Action: show | set"),
    key: Optional[str] = typer.Argument(None, help="Config key (dot-notation, e.g. agent.model)"),
    value: Optional[str] = typer.Argument(None, help="New value to set"),
):
    """Show or update agent configuration."""
    from dev_agent.config import get_settings
    import yaml

    settings = get_settings()

    if action == "show":
        data = {
            "agent": settings.agent.model_dump(),
            "harness": settings.harness.model_dump(),
            "webhooks": settings.webhooks.model_dump(),
            "workspace": settings.workspace_dir,
            "api_key_set": bool(settings.anthropic_api_key),
        }
        console.print(Syntax(yaml.dump(data, default_flow_style=False), "yaml", theme="monokai"))

    elif action == "set":
        if not key or value is None:
            console.print("[red]Usage:[/] dev-agent config set <key> <value>")
            raise typer.Exit(1)
        console.print(f"[yellow]Note:[/] Use .env or config/settings.yaml to persist: [cyan]{key}[/] = [green]{value}[/]")

    else:
        console.print(f"[red]Unknown action:[/] {action}. Use 'show' or 'set'.")
        raise typer.Exit(1)


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


def main():
    app()


if __name__ == "__main__":
    main()
