"""Agent state definition for LangGraph."""

from typing import Annotated, Any
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    workspace: str
    tool_calls_count: int
    interrupted: bool
