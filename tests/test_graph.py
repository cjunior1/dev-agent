"""Integration tests for graph construction — wiring of tools into the ToolNode.

These guard the LLM → graph → ToolNode wiring, which unit tests of the tools
in isolation do not exercise.
"""

from dev_agent.agent.graph import build_graph
from dev_agent.config import get_settings
from dev_agent.tools.registry import build_toolset


class _FakeBoundLLM:
    """Stand-in for an LLM returned by `bind_tools` — like LangChain's
    RunnableBinding, it does NOT expose a `.tools` attribute."""

    def invoke(self, messages):  # pragma: no cover - not exercised here
        raise NotImplementedError


def _tool_names(compiled_graph) -> set[str]:
    """Pull the tool names the compiled graph's ToolNode can actually execute."""
    tool_node = compiled_graph.nodes["tools"].bound
    return set(tool_node.tools_by_name.keys())


def test_graph_tool_node_has_all_tools():
    """The graph's ToolNode must be populated with the tools passed in,
    not silently empty."""
    settings = get_settings()
    tools = build_toolset(None)
    expected = {t.name for t in tools}

    graph = build_graph(_FakeBoundLLM(), settings, tools)
    compiled = graph.compile()

    assert _tool_names(compiled) == expected


def test_graph_tool_node_respects_subset():
    settings = get_settings()
    tools = build_toolset(["shell", "file_read"])

    graph = build_graph(_FakeBoundLLM(), settings, tools)
    compiled = graph.compile()

    assert _tool_names(compiled) == {"shell", "file_read"}
