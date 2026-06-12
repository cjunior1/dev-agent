"""Code quality tools: linting and test runner."""

import asyncio
from pathlib import Path

from langchain_core.tools import tool


async def _run(cmd: str, cwd: str) -> str:
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
        output = stdout.decode(errors="replace")
        if len(output) > 6000:
            output = output[:3000] + "\n...[truncated]...\n" + output[-2000:]
        return output or "(no output)"
    except asyncio.TimeoutError:
        proc.kill()
        return "ERROR: Command timed out after 120s."


@tool
async def code_lint(path: str = ".", linter: str = "ruff") -> str:
    """Run a linter on Python or JS/TS code and return findings.

    Args:
        path: File or directory to lint.
        linter: Linter to use. Supported: 'ruff' (Python), 'eslint' (JS/TS).
    """
    p = Path(path).expanduser().resolve()
    cwd = str(p.parent if p.is_file() else p)

    if linter == "ruff":
        cmd = f"ruff check {path} --output-format=concise"
    elif linter == "eslint":
        cmd = f"npx eslint {path} --format=compact"
    else:
        return f"ERROR: Unknown linter '{linter}'. Supported: ruff, eslint."

    return await _run(cmd, cwd)


@tool
async def test_runner(
    path: str = ".",
    framework: str = "pytest",
    args: str = "-v --tb=short",
) -> str:
    """Run a test suite and return results.

    Args:
        path: Directory or test file to run.
        framework: Test framework. Supported: 'pytest', 'jest', 'vitest', 'go'.
        args: Additional arguments to pass to the test runner.
    """
    p = Path(path).expanduser().resolve()
    cwd = str(p.parent if p.is_file() else p)

    cmd_map = {
        "pytest": f"python -m pytest {path} {args}",
        "jest": f"npx jest {path} {args}",
        "vitest": f"npx vitest run {path} {args}",
        "go": f"go test {args} ./...",
    }

    if framework not in cmd_map:
        return f"ERROR: Unknown framework '{framework}'. Supported: {', '.join(cmd_map)}."

    return await _run(cmd_map[framework], cwd)
