"""File system tools: read, write, list, search.

Reads and searches are unrestricted (the agent legitimately needs to read
system libraries, /etc, code outside the repo, ...). Writes are confined to
the workspace root to prevent the agent from clobbering files outside the
project. The workspace root is held in a ``ContextVar`` so concurrent runs
(e.g. the webhook server) each get their own root without interfering.
"""

from contextvars import ContextVar
from pathlib import Path

from langchain_core.tools import tool

# Workspace root for write confinement. Defaults to the process CWD; the
# harness sets it per run via ``set_workspace_root``.
_workspace_root: ContextVar[str] = ContextVar("cacau_workspace_root", default=".")


def set_workspace_root(path: str) -> None:
    """Set the workspace root that ``file_write`` is confined to (this context)."""
    _workspace_root.set(str(Path(path).expanduser().resolve()))


def _workspace() -> Path:
    return Path(_workspace_root.get()).expanduser().resolve()


def _resolve(path: str) -> Path:
    """Resolve ``path`` to an absolute path. Relative paths resolve against the
    workspace root (so read/write/list/search all interpret them the same way);
    absolute paths are used as-is. No confinement — reads are unrestricted."""
    target = Path(path).expanduser()
    return (target if target.is_absolute() else _workspace() / target).resolve()


def _resolve_within_workspace(path: str) -> Path | None:
    """Resolve ``path`` (see ``_resolve``) and return it only if it stays inside
    the workspace. Returns ``None`` if it would escape. Used to confine writes."""
    root = _workspace()
    target = _resolve(path)
    if target == root or root in target.parents:
        return target
    return None


@tool
def file_read(path: str) -> str:
    """Read the contents of a file.

    Args:
        path: Absolute or relative path to the file.
    """
    p = _resolve(path)
    if not p.exists():
        return f"ERROR: File not found: {path}"
    if p.is_dir():
        return f"ERROR: {path} is a directory. Use file_list instead."
    try:
        content = p.read_text(errors="replace")
        if len(content) > 12000:
            content = content[:6000] + "\n...[truncated - file too large]...\n" + content[-3000:]
        lines = content.splitlines()
        numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
        return numbered
    except Exception as e:
        return f"ERROR: {e}"


@tool
def file_write(path: str, content: str, append: bool = False) -> str:
    """Write or append content to a file, creating parent directories as needed.

    Args:
        path: Path to the file to write.
        content: Text content to write.
        append: If True, append instead of overwrite.
    """
    p = _resolve_within_workspace(path)
    if p is None:
        return f"ERROR: refused to write outside workspace ({_workspace()}): {path}"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        if append:
            with open(p, "a") as f:
                f.write(content)
        else:
            p.write_text(content)
        action = "Appended to" if append else "Wrote"
        return f"{action} {p} ({len(content)} chars)"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def file_list(path: str = ".", pattern: str = "*", recursive: bool = False) -> str:
    """List files and directories at a path, optionally filtering by glob pattern.

    Args:
        path: Directory to list (default: current directory).
        pattern: Glob pattern to filter entries (e.g. '*.py').
        recursive: If True, recurse into subdirectories.
    """
    p = _resolve(path)
    if not p.exists():
        return f"ERROR: Path not found: {path}"
    if not p.is_dir():
        return f"ERROR: {path} is not a directory."

    try:
        glob_fn = p.rglob if recursive else p.glob
        entries = sorted(glob_fn(pattern))
        if not entries:
            return f"(no files matching '{pattern}' in {path})"
        lines = []
        for entry in entries[:200]:
            rel = entry.relative_to(p)
            tag = "/" if entry.is_dir() else ""
            size = entry.stat().st_size if entry.is_file() else 0
            lines.append(f"{'  ' if entry.is_dir() else ''}{rel}{tag}  {size:>8} bytes" if not entry.is_dir() else f"  {rel}{tag}")
        if len(entries) > 200:
            lines.append(f"... and {len(entries)-200} more entries")
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: {e}"


@tool
def code_search(query: str, path: str = ".", extensions: str = "py,js,ts,go,rs,java") -> str:
    """Search for a string/pattern in source code files recursively.

    Args:
        query: Text to search for (case-insensitive).
        path: Root directory to search from.
        extensions: Comma-separated file extensions to include.
    """
    p = _resolve(path)
    exts = {f".{e.strip()}" for e in extensions.split(",")}
    results: list[str] = []

    try:
        for file in p.rglob("*"):
            if file.suffix not in exts or not file.is_file():
                continue
            try:
                text = file.read_text(errors="replace")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if query.lower() in line.lower():
                    rel = file.relative_to(p)
                    results.append(f"{rel}:{i}: {line.rstrip()}")
                    if len(results) >= 100:
                        results.append("...[max results reached]")
                        return "\n".join(results)
        return "\n".join(results) if results else f"No matches for '{query}' in {path}"
    except Exception as e:
        return f"ERROR: {e}"
