"""Smoke tests for dev-agent tools."""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from dev_agent.tools.filesystem import (
    file_read,
    file_write,
    file_list,
    code_search,
    set_workspace_root,
)
from dev_agent.tools.shell import shell
from dev_agent.tools.registry import build_toolset, list_tools


# ── helpers ──────────────────────────────────────────────────────────────────

def run(coro):
    return asyncio.run(coro)


# ── filesystem ───────────────────────────────────────────────────────────────

def test_file_write_and_read(tmp_path):
    set_workspace_root(str(tmp_path))
    target = str(tmp_path / "hello.txt")
    result = file_write.invoke({"path": target, "content": "hello world"})
    assert "Wrote" in result

    content = file_read.invoke({"path": target})
    assert "hello world" in content
    assert "1 |" in content  # line numbers present


def test_file_read_missing():
    result = file_read.invoke({"path": "/nonexistent/path/file.txt"})
    assert "ERROR" in result


def test_file_list(tmp_path):
    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "b.py").write_text("y = 2")
    result = file_list.invoke({"path": str(tmp_path), "pattern": "*.py"})
    assert "a.py" in result
    assert "b.py" in result


def test_code_search(tmp_path):
    (tmp_path / "main.py").write_text("def my_function():\n    return 42\n")
    result = code_search.invoke({"query": "my_function", "path": str(tmp_path)})
    assert "my_function" in result
    assert "main.py" in result


# ── relative paths resolve against the workspace root (read/write parity) ──────

def test_read_resolves_relative_to_workspace(tmp_path):
    set_workspace_root(str(tmp_path))
    (tmp_path / "note.txt").write_text("inside workspace")
    # A relative path must resolve against the workspace root, not the CWD.
    result = file_read.invoke({"path": "note.txt"})
    assert "inside workspace" in result


def test_read_write_same_relative_path_hit_same_file(tmp_path):
    set_workspace_root(str(tmp_path))
    file_write.invoke({"path": "data.txt", "content": "hello"})
    result = file_read.invoke({"path": "data.txt"})
    assert "hello" in result


def test_list_resolves_relative_to_workspace(tmp_path):
    set_workspace_root(str(tmp_path))
    (tmp_path / "a.py").write_text("x = 1")
    result = file_list.invoke({"path": ".", "pattern": "*.py"})
    assert "a.py" in result


# ── shell ─────────────────────────────────────────────────────────────────────

def test_shell_basic():
    result = run(shell.ainvoke({"command": "echo hello_agent"}))
    assert "hello_agent" in result


def test_shell_blocked_command():
    result = run(shell.ainvoke({"command": "rm -rf /"}))
    assert "blocked" in result.lower()


def test_shell_cwd(tmp_path):
    result = run(shell.ainvoke({"command": "pwd", "cwd": str(tmp_path)}))
    assert str(tmp_path) in result


# ── registry ─────────────────────────────────────────────────────────────────

def test_build_toolset_all():
    tools = build_toolset(None)
    assert len(tools) >= 8
    names = {t.name for t in tools}
    assert "shell" in names
    assert "file_read" in names


def test_build_toolset_subset():
    tools = build_toolset(["shell", "file_read"])
    names = {t.name for t in tools}
    assert names == {"shell", "file_read"}


def test_build_toolset_git_group():
    tools = build_toolset(["git"])
    names = {t.name for t in tools}
    assert "git_status" in names
    assert "git_diff" in names


def test_list_tools():
    tools = list_tools()
    assert isinstance(tools, dict)
    assert "shell" in tools
    assert len(tools["shell"]) > 0
