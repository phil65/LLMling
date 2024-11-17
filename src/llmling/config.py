"""Centralized configuration management for LLM services."""

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
import yaml


class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    api_key: str | None = None
    base_url: str | None = None
    models: list[str] = Field(default_factory=list)
    default_model: str | None = None
    organization_id: str | None = None
    timeout: int = 30


class ToolConfig(BaseModel):
    """Configuration for tool usage."""
    enabled_tools: list[str] = Field(default_factory=list)
    auto_tool_choice: bool = True
    tool_definitions_path: Path | None = None


class CacheConfig(BaseModel):
    """Configuration for response caching."""
    enabled: bool = False
    backend: str = "memory"
    ttl: int = 3600
    max_size: int = 1000


class LLMConfig(BaseModel):
    """Main configuration class."""
    providers: dict[str, LLMProviderConfig] = Field(default_factory=dict)
    default_provider: str | None = None
    tools: ToolConfig = Field(default_factory=ToolConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    log_level: str = "INFO"
    temp_dir: Path = Field(default=Path("/tmp"))
    max_retries: int = 3
    fallback_models: list[str] = Field(default_factory=list)


class ConfigManager:
    """Singleton configuration manager."""
    _instance = None
    _config: LLMConfig | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.initialized = True
            self._load_config()

    @property
    def config(self) -> LLMConfig:
        """Get the current configuration."""
        if not self._config:
            self._load_config()
        return self._config

    def _load_config(self) -> None:
        """Load configuration from environment and files."""
        # Default config path locations
        config_paths = [
            Path("config.yaml"),
            Path("~/.jinjarope/config.yaml").expanduser(),
            Path("/etc/jinjarope/config.yaml"),
        ]

        # Environment-specified config path
        if env_path := os.getenv("JINJAROPE_CONFIG"):
            config_paths.insert(0, Path(env_path))

        # Load first existing config file
        config_data = {}
        for path in config_paths:
            if path.is_file():
                with path.open() as f:
                    config_data = yaml.safe_load(f)
                break

        # Override with environment variables
        providers = {}
        for provider in ["openai", "anthropic", "google"]:
            if api_key := os.getenv(f"{provider.upper()}_API_KEY"):
                providers[provider] = LLMProviderConfig(
                    api_key=api_key,
                    base_url=os.getenv(f"{provider.upper()}_BASE_URL"),
                    organization_id=os.getenv(f"{provider.upper()}_ORG_ID")
                )

        # Merge file config with env vars
        if providers:
            config_data["providers"] = {
                **config_data.get("providers", {}),
                **{k: v.model_dump() for k, v in providers.items()}
            }

        self._config = LLMConfig(**config_data)

    def update_config(self, **kwargs: Any) -> None:
        """Update configuration with new values."""
        if not self._config:
            self._load_config()

        config_dict = self._config.model_dump()
        config_dict.update(kwargs)
        self._config = LLMConfig(**config_dict)

    @lru_cache
    def get_provider_config(self, provider: str) -> LLMProviderConfig:
        """Get configuration for a specific provider."""
        if provider not in self.config.providers:
            msg = f"Provider {provider} not configured"
            raise ValueError(msg)
        return self.config.providers[provider]


# Global config instance
config = ConfigManager()


if __name__ == "__main__":
    # Example usage
    print("Default config:", config.config.model_dump_json(indent=2))

    # Example provider config
    try:
        openai_config = config.get_provider_config("openai")
        print("\nOpenAI config:", openai_config.model_dump_json(indent=2))
    except ValueError:
        print("\nOpenAI not configured")

    # Example config update
    config.update_config(
        default_provider="openai",
        providers={
            "openai": {
                "api_key": "test-key",
                "default_model": "gpt-4"
            }
        }
    )
    print("\nUpdated config:", config.config.model_dump_json(indent=2))
