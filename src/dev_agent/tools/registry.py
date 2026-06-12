"""Tool registry — maps names to tool objects and builds enabled toolsets."""

from langchain_core.tools import BaseTool

from dev_agent.tools.shell import shell
from dev_agent.tools.filesystem import file_read, file_write, file_list, code_search
from dev_agent.tools.git_tools import git_status, git_diff, git_commit, git_branch
from dev_agent.tools.code_tools import code_lint, test_runner
from dev_agent.tools.web_tools import web_fetch

_ALL_TOOLS: dict[str, BaseTool] = {
    "shell": shell,
    "file_read": file_read,
    "file_write": file_write,
    "file_list": file_list,
    "code_search": code_search,
    "git_status": git_status,
    "git_diff": git_diff,
    "git_commit": git_commit,
    "git_branch": git_branch,
    "git": git_status,  # alias: 'git' enables all git_* tools
    "code_lint": code_lint,
    "test_runner": test_runner,
    "web_fetch": web_fetch,
}

_GROUPS: dict[str, list[str]] = {
    "git": ["git_status", "git_diff", "git_commit", "git_branch"],
}


def build_toolset(enabled: list[str] | None = None) -> list[BaseTool]:
    """Return the list of enabled tools. If None, return all tools."""
    if enabled is None:
        seen = set()
        result = []
        for tool in _ALL_TOOLS.values():
            if tool.name not in seen:
                seen.add(tool.name)
                result.append(tool)
        return result

    result: list[BaseTool] = []
    seen: set[str] = set()
    for name in enabled:
        # expand groups
        if name in _GROUPS:
            for member in _GROUPS[name]:
                if member not in seen and member in _ALL_TOOLS:
                    result.append(_ALL_TOOLS[member])
                    seen.add(member)
        elif name in _ALL_TOOLS:
            tool = _ALL_TOOLS[name]
            if tool.name not in seen:
                result.append(tool)
                seen.add(tool.name)
    return result


def list_tools() -> dict[str, str]:
    """Return name → description for all available tools."""
    seen: set[str] = set()
    out: dict[str, str] = {}
    for tool in _ALL_TOOLS.values():
        if tool.name not in seen:
            seen.add(tool.name)
            desc = (tool.description or "").split("\n")[0].strip()
            out[tool.name] = desc
    return out
