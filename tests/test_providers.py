"""Tests for the LLM provider factory."""
from unittest.mock import MagicMock, patch

import pytest

from dev_agent.config import LLMProfile
from dev_agent.agent.providers import build_llm, ConfigError


def _profile(**kwargs) -> LLMProfile:
    defaults = dict(provider="anthropic", model="claude-sonnet-4-6",
                    api_key_env="ANTHROPIC_API_KEY", description="test")
    return LLMProfile(**{**defaults, **kwargs})


def test_unknown_provider_raises():
    p = _profile(provider="unknown_provider")
    with pytest.raises(ConfigError, match="Unknown provider"):
        build_llm(p, tools=[])


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = _profile(provider="anthropic", api_key_env="ANTHROPIC_API_KEY")
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        build_llm(p, tools=[])


def test_missing_package_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    p = _profile(provider="openai", api_key_env="OPENAI_API_KEY", model="gpt-4o-mini")
    with patch.dict("sys.modules", {"langchain_openai": None}):
        with pytest.raises(ConfigError, match=r"pip install dev-agent\[openai\]"):
            build_llm(p, tools=[])


def test_anthropic_build(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    p = _profile(provider="anthropic")
    mock_cls = MagicMock()
    mock_cls.return_value.bind_tools.return_value = MagicMock()
    with patch("dev_agent.agent.providers._import_anthropic", return_value=mock_cls):
        build_llm(p, tools=[])
    mock_cls.assert_called_once_with(
        model="claude-sonnet-4-6", temperature=0.1,
        api_key="sk-ant-test", streaming=True,
    )


def test_ollama_skips_api_key_check():
    p = LLMProfile(provider="ollama", model="qwen2.5-coder:7b",
                   base_url="http://localhost:11434", description="local")
    mock_cls = MagicMock()
    mock_cls.return_value.bind_tools.return_value = MagicMock()
    with patch("dev_agent.agent.providers._import_ollama", return_value=mock_cls):
        build_llm(p, tools=[])
    mock_cls.assert_called_once_with(
        model="qwen2.5-coder:7b", base_url="http://localhost:11434"
    )
