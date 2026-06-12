"""File system tools: read, write, list, search."""

import fnmatch
from pathlib import Path

from langchain_core.tools import tool


def _safe_path(path: str, base: str = ".") -> Path:
    base_p = Path(base).expanduser().resolve()
    target = (base_p / path).resolve()
    # Prevent path traversal outside base when base is set explicitly
    return target


@tool
def file_read(path: str) -> str:
    """Read the contents of a file.

    Args:
        path: Absolute or relative path to the file.
    """
    p = _safe_path(path)
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
    p = _safe_path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        p.write_text(content) if not append else open(p, "a").write(content)
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
    p = _safe_path(path)
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
    p = _safe_path(path)
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
