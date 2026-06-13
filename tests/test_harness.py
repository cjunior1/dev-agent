"""Tests for AgentHarness workspace handling across run/resume."""

from pathlib import Path

from dev_agent.agent.harness import AgentHarness
from dev_agent.config import get_settings
from dev_agent.tools.filesystem import _workspace, set_workspace_root


def _harness() -> AgentHarness:
    return AgentHarness(get_settings())


def test_run_workspace_is_remembered_for_resume(tmp_path):
    h = _harness()
    ws = str(tmp_path / "proj")
    h._remember_workspace("tid-1", ws)

    # A resume on the same thread runs in a fresh context (ContextVar back to
    # default); it must restore the workspace recorded by the original run.
    set_workspace_root(".")
    h._restore_workspace("tid-1")
    assert _workspace() == Path(ws).resolve()


def test_resume_unknown_thread_falls_back_to_cwd(tmp_path):
    h = _harness()
    set_workspace_root(str(tmp_path))  # a prior, unrelated run
    h._restore_workspace("never-ran")
    assert _workspace() == Path(".").resolve()
