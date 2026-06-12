"""Tests for the LLM auto-selector."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from dev_agent.config import LLMProfile
from dev_agent.agent.selector import select_profile


PROFILES = {
    "powerful": LLMProfile(provider="anthropic", model="claude-opus-4-8",
                            description="Best for complex reasoning and architecture."),
    "fast": LLMProfile(provider="groq", model="llama-3.3-70b-versatile",
                       description="Fast and cheap for simple tasks."),
}


@pytest.mark.asyncio
async def test_select_returns_valid_profile():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="powerful"))

    result = await select_profile("refactor the payments module", PROFILES, mock_llm)
    assert result == "powerful"


@pytest.mark.asyncio
async def test_select_strips_whitespace_and_quotes():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content='  "fast"  '))

    result = await select_profile("explain this function quickly", PROFILES, mock_llm)
    assert result == "fast"


@pytest.mark.asyncio
async def test_select_falls_back_on_unknown_profile():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="nonexistent_profile"))

    result = await select_profile("do something", PROFILES, mock_llm)
    assert result == "powerful"  # first profile in dict


@pytest.mark.asyncio
async def test_select_falls_back_on_exception():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("network error"))

    result = await select_profile("do something", PROFILES, mock_llm)
    assert result == "powerful"  # first profile
