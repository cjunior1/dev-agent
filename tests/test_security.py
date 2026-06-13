"""Security tests: shell-injection in git tools, write confinement, shell blocklist."""

import asyncio
import subprocess

from dev_agent.tools.git_tools import git_commit, git_branch
from dev_agent.tools.filesystem import file_write, set_workspace_root
from dev_agent.tools.shell import _is_safe


def run(coro):
    return asyncio.run(coro)


def _init_repo(path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)


# ── git command injection ────────────────────────────────────────────────────

def test_git_commit_no_shell_injection(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("x")
    marker = tmp_path / "INJECTED"

    # A message crafted to break out of the shell command and run `touch INJECTED`.
    run(git_commit.ainvoke({
        "message": 'msg"; touch INJECTED; echo "',
        "cwd": str(tmp_path),
        "add_all": True,
    }))

    assert not marker.exists(), "shell injection executed via commit message"


def test_git_branch_no_shell_injection(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("x")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    marker = tmp_path / "INJECTED"

    run(git_branch.ainvoke({
        "cwd": str(tmp_path),
        "create": "foo; touch INJECTED",
    }))

    assert not marker.exists(), "shell injection executed via branch name"


# ── write confinement ────────────────────────────────────────────────────────

def test_file_write_allows_inside_workspace(tmp_path):
    set_workspace_root(str(tmp_path))
    target = str(tmp_path / "sub" / "ok.txt")
    result = file_write.invoke({"path": target, "content": "hi"})
    assert "Wrote" in result
    assert (tmp_path / "sub" / "ok.txt").read_text() == "hi"


def test_file_write_refuses_outside_workspace(tmp_path):
    set_workspace_root(str(tmp_path / "workspace"))
    (tmp_path / "workspace").mkdir()
    outside = str(tmp_path / "escape.txt")
    result = file_write.invoke({"path": outside, "content": "pwned"})
    assert "ERROR" in result
    assert not (tmp_path / "escape.txt").exists()


def test_file_write_refuses_traversal(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    set_workspace_root(str(ws))
    result = file_write.invoke({"path": "../escape.txt", "content": "pwned"})
    assert "ERROR" in result
    assert not (tmp_path / "escape.txt").exists()


# ── shell blocklist robustness ───────────────────────────────────────────────

def test_shell_blocklist_normalizes_whitespace():
    # Extra spaces must not slip past the blocklist.
    assert _is_safe("rm  -rf   /") is False
    assert _is_safe("RM -RF /") is False
    assert _is_safe("echo hello") is True
