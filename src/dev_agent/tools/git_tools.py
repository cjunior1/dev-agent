"""Git integration tools."""

import asyncio

from langchain_core.tools import tool


async def _git(args: list[str], cwd: str = ".") -> str:
    # Use exec (not shell) with an explicit argv so user-supplied values
    # (commit messages, branch names, refs) cannot inject shell commands.
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    return stdout.decode(errors="replace").strip()


@tool
async def git_status(cwd: str = ".") -> str:
    """Show git working tree status and recent log.

    Args:
        cwd: Repository directory (default: current).
    """
    status = await _git(["status", "--short", "--branch"], cwd)
    log = await _git(["log", "--oneline", "-10"], cwd)
    return f"=== STATUS ===\n{status}\n\n=== RECENT LOG ===\n{log}"


@tool
async def git_diff(cwd: str = ".", ref: str = "HEAD") -> str:
    """Show git diff against a reference (default: unstaged changes vs HEAD).

    Args:
        cwd: Repository directory.
        ref: Git ref to diff against (e.g. 'HEAD', 'main', 'HEAD~1').
    """
    result = await _git(["diff", ref], cwd)
    if len(result) > 8000:
        result = result[:8000] + "\n...[diff truncated]"
    return result or "(no changes)"


@tool
async def git_commit(message: str, cwd: str = ".", add_all: bool = False) -> str:
    """Create a git commit.

    Args:
        message: Commit message.
        cwd: Repository directory.
        add_all: If True, stage all changes before committing (git add -A).
    """
    if add_all:
        await _git(["add", "-A"], cwd)
    return await _git(["commit", "-m", message], cwd)


@tool
async def git_branch(cwd: str = ".", create: str = "", checkout: str = "") -> str:
    """List, create, or checkout git branches.

    Args:
        cwd: Repository directory.
        create: Branch name to create (and checkout).
        checkout: Branch name to checkout (must already exist).
    """
    if create:
        return await _git(["checkout", "-b", create], cwd)
    if checkout:
        return await _git(["checkout", checkout], cwd)
    return await _git(["branch", "-a"], cwd)
