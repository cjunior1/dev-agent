"""LangGraph Deep Agent graph construction."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from dev_agent.agent.state import AgentState
from dev_agent.agent.prompts import build_system_prompt
from dev_agent.config import Settings


def build_graph(llm: BaseChatModel, settings: Settings, tools: list[BaseTool]):
    """Build and compile the deep agent StateGraph.

    Accepts a BaseChatModel already bound to tools (via providers.build_llm)
    and the original tool list to populate the ToolNode. The bound LLM does
    not expose the original tool objects (``bind_tools`` returns a
    ``RunnableBinding`` whose tools live as schema dicts in ``.kwargs``), so
    the executable tools must be passed in explicitly.

    Implements a ReAct loop with a max-iterations guard.
    """
    max_iter = settings.agent.max_iterations

    def agent_node(state: AgentState) -> dict:
        system_msg = SystemMessage(content=build_system_prompt(state.get("workspace", ".")))
        messages = [system_msg] + list(state["messages"])
        response = llm.invoke(messages)
        count = state.get("tool_calls_count", 0)
        if hasattr(response, "tool_calls") and response.tool_calls:
            count += len(response.tool_calls)
        return {"messages": [response], "tool_calls_count": count}

    def should_continue(state: AgentState) -> str:
        if state.get("tool_calls_count", 0) >= max_iter:
            return END
        return tools_condition(state)

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph
