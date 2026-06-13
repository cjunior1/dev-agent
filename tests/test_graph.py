"""Integration tests for graph construction.

These exercise the LLM → graph → ToolNode wiring by actually running the graph
with a scripted LLM that emits a tool call, asserting the tool *executes*. This
guards the original bug (the ToolNode was built empty, so every tool call
errored at runtime) through observable behavior rather than internal structure.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from dev_agent.agent.graph import build_graph
from dev_agent.config import get_settings
from dev_agent.tools.filesystem import set_workspace_root
from dev_agent.tools.registry import build_toolset


class _ScriptedLLM:
    """Stands in for a bind_tools'd model: emits one tool call, then finishes.

    Like LangChain's RunnableBinding it exposes no `.tools` attribute, so it
    also proves the graph does not rely on one."""

    def __init__(self, tool_name: str, tool_args: dict):
        self._tool_name = tool_name
        self._tool_args = tool_args

    def invoke(self, messages):
        # Once the tool has run (last message is its result), stop.
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="done")
        return AIMessage(
            content="",
            tool_calls=[{"name": self._tool_name, "args": self._tool_args, "id": "call-1"}],
        )


def _run_graph(tools, tool_name, tool_args, workspace="."):
    llm = _ScriptedLLM(tool_name, tool_args)
    compiled = build_graph(llm, get_settings(), tools).compile()
    state = compiled.invoke({
        "messages": [HumanMessage(content="go")],
        "workspace": workspace,
        "tool_calls_count": 0,
        "interrupted": False,
    })
    return state["messages"]


def _tool_messages(messages):
    return [m for m in messages if isinstance(m, ToolMessage)]


def test_tool_call_executes_through_graph():
    """A tool the model calls must actually run — not error on an empty ToolNode."""

    @tool
    def echo(text: str) -> str:
        """Echo the given text back."""
        return f"echoed:{text}"

    messages = _run_graph([echo], "echo", {"text": "hi"})

    outputs = [m.content for m in _tool_messages(messages)]
    assert "echoed:hi" in outputs


def test_registered_tools_execute_through_graph(tmp_path):
    """The real registry toolset is wired and executable end-to-end."""
    set_workspace_root(str(tmp_path))
    tools = build_toolset(None)

    messages = _run_graph(
        tools, "file_write", {"path": "out.txt", "content": "hello"}, workspace=str(tmp_path)
    )

    assert (tmp_path / "out.txt").read_text() == "hello"
    assert any("Wrote" in m.content for m in _tool_messages(messages))
