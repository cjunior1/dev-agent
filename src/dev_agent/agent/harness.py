"""AgentHarness — wires checkpointing, interrupt hooks, streaming, and LLM selection."""

import uuid
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from dev_agent.agent.graph import build_graph
from dev_agent.agent.providers import build_llm
from dev_agent.agent.selector import select_profile
from dev_agent.config import LLMProfile, Settings, get_settings
from dev_agent.tools.filesystem import set_workspace_root
from dev_agent.tools.registry import build_toolset


class AgentHarness:
    """Manages the compiled agent graph with harness features:
    - MemorySaver checkpointer (thread-scoped conversation memory)
    - Configurable interrupt_before / interrupt_after hooks
    - Multi-LLM profile support with auto-selection
    - Streaming event emission with profile_selected events
    - Human-in-the-loop resume support
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._checkpointer = MemorySaver()
        self._tools = build_toolset(None)
        # Per-thread workspace root, so a resumed conversation confines writes
        # to the same workspace as its original run. Shares the in-memory
        # lifetime of the checkpointer.
        self._workspaces: dict[str, str] = {}

    def new_thread(self) -> str:
        return str(uuid.uuid4())

    def _remember_workspace(self, thread_id: str, workspace: str) -> None:
        """Record a run's workspace and confine writes to it (this context)."""
        self._workspaces[thread_id] = workspace
        set_workspace_root(workspace)

    def _restore_workspace(self, thread_id: str) -> None:
        """Re-confine writes to the workspace of a thread's original run,
        falling back to the process CWD for unknown threads."""
        set_workspace_root(self._workspaces.get(thread_id, "."))

    def _run_config(self, thread_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.settings.agent.recursion_limit,
        }

    async def _resolve_profile(self, prompt: str, override: str | None) -> tuple[str, LLMProfile]:
        """Return (profile_name, profile) — runs auto-selection if needed."""
        requested = override or self.settings.agent.profile

        if requested != "auto" and requested in self.settings.profiles:
            return requested, self.settings.profiles[requested]

        # auto mode: use classifier LLM to pick
        selector_profile_name = self.settings.llm_selector.profile
        selector_profile = self.settings.profiles.get(
            selector_profile_name, next(iter(self.settings.profiles.values()))
        )
        classifier_llm = build_llm(selector_profile, tools=[])
        chosen_name = await select_profile(prompt, self.settings.profiles, classifier_llm)
        return chosen_name, self.settings.profiles[chosen_name]

    async def run(
        self,
        prompt: str,
        thread_id: str | None = None,
        workspace: str = ".",
        profile: str | None = None,
        model: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream agent events for a given prompt.

        Yields event dicts with keys: type, payload, thread_id.
        Event types: 'profile_selected' | 'token' | 'tool_call' | 'tool_result' | 'done'
        """
        thread_id = thread_id or self.new_thread()
        self._remember_workspace(thread_id, workspace)

        profile_name, selected_profile = await self._resolve_profile(prompt, profile)
        if model:
            selected_profile = selected_profile.model_copy(update={"model": model})
        yield {
            "type": "profile_selected",
            "payload": {"name": profile_name, "model": selected_profile.model,
                        "provider": selected_profile.provider},
            "thread_id": thread_id,
        }

        llm = build_llm(selected_profile, self._tools)
        graph = build_graph(llm, self.settings, self._tools)
        compiled = graph.compile(
            checkpointer=self._checkpointer,
            interrupt_before=self.settings.harness.interrupt_before or [],
            interrupt_after=self.settings.harness.interrupt_after or [],
        )

        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "workspace": workspace,
            "tool_calls_count": 0,
            "interrupted": False,
        }

        async for event in compiled.astream_events(
            initial_state, config=self._run_config(thread_id), version="v2"
        ):
            kind = event.get("event", "")
            name = event.get("name", "")
            data = event.get("data", {})

            if kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield {"type": "token", "payload": chunk.content, "thread_id": thread_id}

            elif kind == "on_tool_start":
                yield {"type": "tool_call",
                       "payload": {"tool": name, "input": data.get("input", {})},
                       "thread_id": thread_id}

            elif kind == "on_tool_end":
                yield {"type": "tool_result",
                       "payload": {"tool": name, "output": str(data.get("output", ""))[:2000]},
                       "thread_id": thread_id}

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
        profile: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Resume a graph that was interrupted (human-in-the-loop)."""
        self._restore_workspace(thread_id)
        profile_name, selected_profile = await self._resolve_profile("", profile)
        llm = build_llm(selected_profile, self._tools)
        graph = build_graph(llm, self.settings, self._tools)
        compiled = graph.compile(
            checkpointer=self._checkpointer,
            interrupt_before=self.settings.harness.interrupt_before or [],
            interrupt_after=self.settings.harness.interrupt_after or [],
        )

        async for event in compiled.astream_events(
            value, config=self._run_config(thread_id), version="v2"
        ):
            kind = event.get("event", "")
            data = event.get("data", {})
            name = event.get("name", "")

            if kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield {"type": "token", "payload": chunk.content, "thread_id": thread_id}

            elif kind == "on_tool_start":
                yield {"type": "tool_call",
                       "payload": {"tool": name, "input": data.get("input", {})},
                       "thread_id": thread_id}

            elif kind == "on_tool_end":
                yield {"type": "tool_result",
                       "payload": {"tool": name, "output": str(data.get("output", ""))[:2000]},
                       "thread_id": thread_id}

            elif kind == "on_chain_end" and name == "LangGraph":
                output = data.get("output", {})
                messages = output.get("messages", [])
                last = messages[-1] if messages else None
                yield {"type": "done", "payload": getattr(last, "content", ""), "thread_id": thread_id}

    def get_state(self, thread_id: str):
        return self._checkpointer

    def list_threads(self) -> list[str]:
        try:
            return [c.config["configurable"]["thread_id"] for c in self._checkpointer.list({})]
        except Exception:
            return []
