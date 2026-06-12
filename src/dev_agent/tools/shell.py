"""Shell execution tool with safety constraints."""

import asyncio
import shlex
from pathlib import Path

from langchain_core.tools import tool


BLOCKED_COMMANDS = frozenset([
    "rm -rf /", "mkfs", "dd if=", ":(){:|:&};:",
    "chmod -R 777 /", "chown -R",
])


def _is_safe(command: str) -> bool:
    cmd_lower = command.lower().strip()
    return not any(blocked in cmd_lower for blocked in BLOCKED_COMMANDS)


@tool
async def shell(command: str, cwd: str = ".") -> str:
    """Execute a shell command in the workspace. Returns stdout+stderr combined.

    Args:
        command: Shell command to run (bash syntax).
        cwd: Working directory relative to workspace (default: current dir).
    """
    if not _is_safe(command):
        return "ERROR: Command blocked by safety policy."

    work_dir = Path(cwd).expanduser().resolve()
    if not work_dir.exists():
        work_dir = Path(".")

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(work_dir),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        output = stdout.decode(errors="replace")
        if len(output) > 8000:
            output = output[:4000] + "\n...[truncated]...\n" + output[-2000:]
        return output or "(no output)"
    except asyncio.TimeoutError:
        proc.kill()
        return "ERROR: Command timed out after 60s."
    except Exception as e:
        return f"ERROR: {e}"
