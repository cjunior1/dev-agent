"""AgentHarness — wires checkpointing, interrupt hooks, and streaming."""

import uuid
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from dev_agent.agent.graph import build_graph
from dev_agent.config import Settings, get_settings
from dev_agent.tools.registry import build_toolset


class AgentHarness:
    """Manages the compiled agent graph with harness features:
    - MemorySaver checkpointer (thread-scoped conversation memory)
    - Configurable interrupt_before / interrupt_after hooks
    - Streaming event emission
    - Human-in-the-loop resume support
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._checkpointer = MemorySaver()
        self._tools = build_toolset(None)  # load all; CLI/config can filter
        graph = build_graph(self._tools, self.settings)
        self._graph = graph.compile(
            checkpointer=self._checkpointer,
            interrupt_before=self.settings.harness.interrupt_before or [],
            interrupt_after=self.settings.harness.interrupt_after or [],
        )

    def new_thread(self) -> str:
        return str(uuid.uuid4())

    def _config(self, thread_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.settings.agent.recursion_limit,
        }

    async def run(
        self,
        prompt: str,
        thread_id: str | None = None,
        workspace: str = ".",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream agent events for a given prompt.

        Yields dicts with keys: type ('token'|'tool_call'|'tool_result'|'done'), payload.
        """
        thread_id = thread_id or self.new_thread()
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "workspace": workspace,
            "tool_calls_count": 0,
            "interrupted": False,
        }

        async for event in self._graph.astream_events(
            initial_state,
            config=self._config(thread_id),
            version="v2",
        ):
            kind = event.get("event", "")
            name = event.get("name", "")
            data = event.get("data", {})

            if kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield {"type": "token", "payload": chunk.content, "thread_id": thread_id}

            elif kind == "on_tool_start":
                yield {
                    "type": "tool_call",
                    "payload": {"tool": name, "input": data.get("input", {})},
                    "thread_id": thread_id,
                }

            elif kind == "on_tool_end":
                yield {
                    "type": "tool_result",
                    "payload": {"tool": name, "output": str(data.get("output", ""))[:2000]},
                    "thread_id": thread_id,
                }

            elif kind == "on_chain_end" and name == "LangGraph":
                output = data.get("output", {})
                messages = output.get("messages", [])
                last = messages[-1] if messages else None
                final_text = getattr(last, "content", "") if last else ""
                yield {"type": "done", "payload": final_text, "thread_id": thread_id}

    async def resume(
        self,
        thread_id: str,
        value: Any = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Resume a graph that was interrupted (human-in-the-loop)."""
        async for event in self._graph.astream_events(
            value,
            config=self._config(thread_id),
            version="v2",
        ):
            kind = event.get("event", "")
            data = event.get("data", {})
            name = event.get("name", "")

            if kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield {"type": "token", "payload": chunk.content, "thread_id": thread_id}

            elif kind == "on_tool_start":
                yield {"type": "tool_call", "payload": {"tool": name, "input": data.get("input", {})}, "thread_id": thread_id}

            elif kind == "on_tool_end":
                yield {"type": "tool_result", "payload": {"tool": name, "output": str(data.get("output", ""))[:2000]}, "thread_id": thread_id}

            elif kind == "on_chain_end" and name == "LangGraph":
                output = data.get("output", {})
                messages = output.get("messages", [])
                last = messages[-1] if messages else None
                yield {"type": "done", "payload": getattr(last, "content", ""), "thread_id": thread_id}

    def get_state(self, thread_id: str):
        """Return the current checkpoint state for a thread."""
        return self._graph.get_state(self._config(thread_id))

    def list_threads(self) -> list[str]:
        """Return thread IDs stored in the checkpointer."""
        try:
            return [c.config["configurable"]["thread_id"] for c in self._checkpointer.list({})]
        except Exception:
            return []
