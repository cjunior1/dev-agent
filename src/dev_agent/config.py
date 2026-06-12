"""Configuration management with pydantic-settings and YAML support."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_sub_config = SettingsConfigDict(extra="ignore")


class AgentConfig(BaseSettings):
    model_config = _sub_config
    model: str = "claude-sonnet-4-6"
    max_iterations: int = 25
    recursion_limit: int = 50
    temperature: float = 0.1
    streaming: bool = True


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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DEV_AGENT_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    agent: AgentConfig = Field(default_factory=AgentConfig)
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    webhooks: WebhookConfig = Field(default_factory=WebhookConfig)
    workspace_dir: str = Field(default=".", alias="DEV_AGENT_WORKSPACE")

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        if not path.exists():
            return cls()
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        agent_data = data.pop("agent", {})
        harness_data = data.pop("harness", {})
        webhooks_data = data.pop("webhooks", {})
        data.pop("tools", None)
        data.pop("cli", None)
        return cls(
            agent=AgentConfig(**agent_data),
            harness=HarnessConfig(**harness_data),
            webhooks=WebhookConfig(**webhooks_data),
            **data,
        )


_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml(_CONFIG_PATH)
        # env var override for API key
        if not _settings.anthropic_api_key:
            _settings.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    return _settings
