from __future__ import annotations

import pytest

from llmling import config_resources
from llmling.config.manager import ConfigManager
from llmling.config.models import Config, GlobalSettings, TextResource, ToolConfig
from llmling.config.runtime import RuntimeConfig
from llmling.processors.registry import ProcessorRegistry
from llmling.prompts.models import PromptMessage, StaticPrompt
from llmling.resources.loaders import ResourceLoaderRegistry


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
def loader_registry() -> ResourceLoaderRegistry:
    """Create a populated resource registry."""
    registry = ResourceLoaderRegistry()
    registry.register_default_loaders()
    return registry


@pytest.fixture
def test_config() -> Config:
    """Create test configuration with sample data."""
    prompt_msg = PromptMessage(role="user", content="Test message")
    text = TextResource(content="test content", description="Test resource")
    path = "llmling.testing.tools.example_tool"
    desc = "Example tool for testing"
    cfg = ToolConfig(import_path=path, name="example", description=desc)
    prompt = StaticPrompt(name="test-prompt", description="Test", messages=[prompt_msg])
    return Config(
        global_settings=GlobalSettings(),
        resources={"test-resource": text},
        tools={"example": cfg},
        prompts={"test-prompt": prompt},
    )


@pytest.fixture
def runtime_config(test_config: Config) -> RuntimeConfig:
    """Create test runtime configuration."""
    return RuntimeConfig.from_config(test_config)


@pytest.fixture
def runtime() -> RuntimeConfig:
    """Fixture providing a RuntimeConfig."""
    from llmling.config.models import Config

    config = Config.from_file(config_resources.TEST_CONFIG)
    return RuntimeConfig.from_config(config)
