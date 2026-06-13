"""Interactive REPL for the dev agent with Rich rendering."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

if TYPE_CHECKING:
    from dev_agent.agent.harness import AgentHarness

console = Console()

SLASH_COMMANDS = {
    "/help": "Show available slash commands",
    "/tools": "List all available tools",
    "/history": "Show session history",
    "/clear": "Clear the screen",
    "/thread": "Show current thread ID",
    "/threads": "List all thread IDs",
    "/exit": "Exit the REPL",
}

HISTORY_FILE = Path("~/.cacau_history").expanduser()


def _load_history() -> list[str]:
    if HISTORY_FILE.exists():
        return HISTORY_FILE.read_text().splitlines()[-500:]
    return []


def _save_history(history: list[str]) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text("\n".join(history[-500:]))


def _render_tool_call(name: str, inputs: dict) -> None:
    console.print(f"\n[bold cyan]⚙ Tool:[/] [cyan]{name}[/]")
    if inputs:
        try:
            body = json.dumps(inputs, indent=2)
            console.print(Syntax(body, "json", theme="monokai", word_wrap=True))
        except Exception:
            console.print(Text(str(inputs), style="dim"))


def _render_tool_result(name: str, output: str) -> None:
    trimmed = output[:800] + ("…" if len(output) > 800 else "")
    console.print(Panel(trimmed, title=f"[dim]{name} result[/dim]", border_style="dim", expand=False))


async def _stream_response(
    harness: "AgentHarness",
    prompt: str,
    thread_id: str,
    workspace: str,
    default_profile: str | None,
    default_model: str | None = None,
) -> str:
    """Stream the agent response, rendering events as they arrive."""
    full_response = ""
    is_auto = default_profile is None and harness.settings.agent.profile == "auto"
    console.print()

    async for event in harness.run(prompt, thread_id=thread_id, workspace=workspace, profile=default_profile, model=default_model):
        etype = event["type"]
        payload = event["payload"]

        if etype == "profile_selected" and is_auto:
            name = payload["name"]
            model = payload["model"]
            console.print(f"[dim][auto → [cyan]{name}[/cyan] · {model}][/dim]")

        elif etype == "token":
            console.print(payload, end="", highlight=False)
            full_response += payload

        elif etype == "tool_call":
            _render_tool_call(payload["tool"], payload.get("input", {}))

        elif etype == "tool_result":
            _render_tool_result(payload["tool"], payload.get("output", ""))

        elif etype == "done":
            if not full_response and payload:
                console.print(Markdown(payload))
                full_response = payload

    console.print()
    return full_response


async def run_repl(harness: "AgentHarness", workspace: str = ".", default_profile: str | None = None, default_model: str | None = None) -> None:
    """Start the interactive REPL loop."""
    from dev_agent.tools.registry import list_tools

    thread_id = harness.new_thread()
    history: list[str] = _load_history()

    console.print(Panel(
        "[bold green]Cacau[/bold green] — interactive mode\n"
        "[dim]Type your request or a /command. Use /help to see commands.[/dim]",
        border_style="green",
    ))
    console.print(f"[dim]Thread: {thread_id}  |  Workspace: {workspace}[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold blue]you>[/bold blue] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            break

        if not user_input:
            continue

        history.append(user_input)

        # --- slash commands ---
        if user_input.startswith("/"):
            cmd = user_input.split()[0]

            if cmd in ("/exit", "/quit", "/q"):
                console.print("[dim]Bye![/dim]")
                break

            elif cmd == "/clear":
                console.clear()

            elif cmd == "/help":
                for c, desc in SLASH_COMMANDS.items():
                    console.print(f"  [cyan]{c:<12}[/]  {desc}")

            elif cmd == "/tools":
                tools = list_tools()
                for name, desc in tools.items():
                    console.print(f"  [cyan]{name:<18}[/]  {desc}")

            elif cmd == "/history":
                for i, h in enumerate(history[-20:], 1):
                    console.print(f"  [dim]{i:2}.[/] {h}")

            elif cmd == "/thread":
                console.print(f"[dim]Current thread: {thread_id}[/dim]")

            elif cmd == "/threads":
                threads = harness.list_threads()
                if threads:
                    for t in threads:
                        mark = " [bold green]<current>[/]" if t == thread_id else ""
                        console.print(f"  [dim]{t}[/dim]{mark}")
                else:
                    console.print("[dim]No threads stored.[/dim]")

            elif cmd == "/new":
                thread_id = harness.new_thread()
                console.print(f"[dim]New thread: {thread_id}[/dim]")

            else:
                console.print(f"[red]Unknown command:[/] {cmd}. Type /help for help.")
            continue

        # --- agent invocation ---
        try:
            await _stream_response(harness, user_input, thread_id, workspace, default_profile, default_model)
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")

    _save_history(history)
