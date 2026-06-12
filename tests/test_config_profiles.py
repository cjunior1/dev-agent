"""Tests for LLMProfile config parsing."""
from dev_agent.config import LLMProfile, LLMSelectorConfig, Settings


def test_llm_profile_fields():
    p = LLMProfile(
        provider="anthropic",
        model="claude-sonnet-4-6",
        api_key_env="ANTHROPIC_API_KEY",
        temperature=0.1,
        streaming=True,
        description="Test profile",
    )
    assert p.provider == "anthropic"
    assert p.api_key_env == "ANTHROPIC_API_KEY"
    assert p.base_url is None


def test_llm_profile_ollama_base_url():
    p = LLMProfile(
        provider="ollama",
        model="qwen2.5-coder:7b",
        base_url="http://localhost:11434",
        description="Local model",
    )
    assert p.base_url == "http://localhost:11434"
    assert p.api_key_env is None


def test_settings_has_profiles():
    s = Settings(profiles={
        "fast": LLMProfile(provider="groq", model="llama-3.3-70b-versatile", description="Fast")
    })
    assert "fast" in s.profiles
    assert s.profiles["fast"].provider == "groq"


def test_settings_default_profile_is_auto():
    s = Settings()
    assert s.agent.profile == "auto"


def test_llm_selector_config():
    c = LLMSelectorConfig(profile="fast")
    assert c.profile == "fast"
