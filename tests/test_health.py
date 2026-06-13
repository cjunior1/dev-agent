"""Tests for profile health checker."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dev_agent.config import LLMProfile
from dev_agent.agent.health import check_profile, check_all


PROFILES = {
    "fast": LLMProfile(provider="groq", model="llama-3.3-70b-versatile",
                       api_key_env="GROQ_API_KEY", description="Fast"),
    "local": LLMProfile(provider="ollama", model="qwen2.5-coder:7b",
                        base_url="http://localhost:11434", description="Local"),
}


@pytest.mark.asyncio
async def test_check_profile_ok(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Paris"))

    with patch("dev_agent.agent.health.build_llm", return_value=mock_llm):
        status = await check_profile("fast", PROFILES["fast"])

    assert status.name == "fast"
    assert status.ok is True
    assert "Paris" in status.snippet
    assert status.error is None


@pytest.mark.asyncio
async def test_check_profile_missing_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    status = await check_profile("fast", PROFILES["fast"])

    assert status.ok is False
    assert "GROQ_API_KEY" in status.error


@pytest.mark.asyncio
async def test_check_profile_llm_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("connection refused"))

    with patch("dev_agent.agent.health.build_llm", return_value=mock_llm):
        status = await check_profile("fast", PROFILES["fast"])

    assert status.ok is False
    assert "connection refused" in status.error


@pytest.mark.asyncio
async def test_check_all_returns_all_profiles():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Paris"))

    with patch("dev_agent.agent.health.build_llm", return_value=mock_llm):
        results = await check_all(PROFILES)

    assert len(results) == 2
    names = [r.name for r in results]
    assert "fast" in names
    assert "local" in names
