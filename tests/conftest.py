from __future__ import annotations

import pytest

from llmling.config.manager import ConfigManager
from llmling.config.models import Config, GlobalSettings, TextResource, ToolConfig
from llmling.config.runtime import RuntimeConfig
from llmling.processors.registry import ProcessorRegistry
from llmling.prompts.models import Prompt, PromptMessage


@pytest.fixture(autouse=True)
def disable_logfire():
    """Disable Logfire for all tests."""
    import os

    os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"


@pytest.fixture
def config_manager(test_config):
    """Get config manager with test configuration."""
    return ConfigManager(test_config)


@pytest.fixture
def processor_registry():
    """Get clean processor registry."""
    return ProcessorRegistry()


@pytest.fixture
def test_config() -> Config:
    """Create test configuration with sample data."""
    prompt_msg = PromptMessage(role="user", content="Test message")
    return Config(
        version="1.0.0",
        global_settings=GlobalSettings(),
        resources={
            "test-resource": TextResource(
                content="test content", description="Test resource"
            ),
        },
        tools={
            "example": ToolConfig(
                import_path="llmling.testing.tools.example_tool",
                name="example",
                description="Example tool for testing",
            ),
        },
        prompts={
            "test-prompt": Prompt(
                name="test-prompt",
                description="Test prompt",
                messages=[prompt_msg],
            ),
        },
    )


@pytest.fixture
def runtime_config(test_config: Config) -> RuntimeConfig:
    """Create test runtime configuration."""
    return RuntimeConfig.from_config(test_config)
