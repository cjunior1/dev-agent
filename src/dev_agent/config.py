"""Configuration management with pydantic-settings and YAML support."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_sub_config = SettingsConfigDict(extra="ignore")


class LLMProfile(BaseModel):
    """A named LLM configuration profile."""
    model_config = SettingsConfigDict(extra="ignore")

    provider: str                        # anthropic | openai | google | ollama | groq
    model: str
    description: str = ""
    api_key_env: str | None = None       # env var name holding the API key
    base_url: str | None = None          # for ollama or custom endpoints
    temperature: float = 0.1
    streaming: bool = True


class LLMSelectorConfig(BaseModel):
    model_config = SettingsConfigDict(extra="ignore")
    profile: str = "fast"               # profile name to use as classifier


class AgentConfig(BaseSettings):
    model_config = _sub_config
    profile: str = "auto"               # profile name or "auto"
    max_iterations: int = 25
    recursion_limit: int = 50
    streaming: bool = True              # fallback default; profile.streaming takes precedence


class HarnessConfig(BaseSettings):
    model_config = _sub_config
    checkpointing: bool = True
    interrupt_before: list[str] = Field(default_factory=list)
    interrupt_after: list[str] = Field(default_factory=list)
    debug_mode: bool = False


class WebhookConfig(BaseSettings):
    model_config = _sub_config
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    secret: str = ""


_DEFAULT_PROFILES: dict[str, LLMProfile] = {
    "default": LLMProfile(
        provider="anthropic",
        model="claude-sonnet-4-6",
        api_key_env="ANTHROPIC_API_KEY",
        description="Default Anthropic Claude profile for general development tasks.",
    )
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CACAU_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    profiles: dict[str, LLMProfile] = Field(default_factory=lambda: dict(_DEFAULT_PROFILES))
    llm_selector: LLMSelectorConfig = Field(default_factory=LLMSelectorConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    webhooks: WebhookConfig = Field(default_factory=WebhookConfig)
    workspace_dir: str = Field(default=".", alias="CACAU_WORKSPACE")

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        if not path.exists():
            return cls()
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        raw_profiles = data.pop("profiles", {})
        profiles = {name: LLMProfile(**cfg) for name, cfg in raw_profiles.items()} if raw_profiles else dict(_DEFAULT_PROFILES)

        raw_selector = data.pop("llm_selector", {})
        llm_selector = LLMSelectorConfig(**raw_selector) if raw_selector else LLMSelectorConfig()

        agent_data = data.pop("agent", {})
        harness_data = data.pop("harness", {})
        webhooks_data = data.pop("webhooks", {})
        data.pop("tools", None)
        data.pop("cli", None)

        return cls(
            profiles=profiles,
            llm_selector=llm_selector,
            agent=AgentConfig(**agent_data),
            harness=HarnessConfig(**harness_data),
            webhooks=WebhookConfig(**webhooks_data),
            **data,
        )

    def get_profile(self, name: str | None = None) -> LLMProfile:
        """Return the named profile, or the agent's configured profile."""
        key = name or self.agent.profile
        if key == "auto" or key not in self.profiles:
            return next(iter(self.profiles.values()))
        return self.profiles[key]


_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml(_CONFIG_PATH)
    return _settings


def reset_settings() -> None:
    """Force reload on next get_settings() call. Used in tests."""
    global _settings
    _settings = None
