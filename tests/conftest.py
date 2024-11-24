from __future__ import annotations

from typing import Any

import pytest

from llmling.client import LLMLingClient
from llmling.config.manager import ConfigManager
from llmling.config.models import (
    Config,
    GlobalSettings,
    LLMProviderConfig,
    TaskTemplate,
    TextContext,
)
from llmling.context import default_registry as context_registry
from llmling.llm.registry import default_registry as llm_registry
from llmling.processors.registry import ProcessorRegistry
from llmling.task.executor import TaskExecutor
from llmling.task.manager import TaskManager


# Standard test responses
DEFAULT_TEST_RESPONSES = {
    "default": {
        "content": "Test response content",
        "model": "test/model",
        "metadata": {"test": "metadata"},
    },
    "Hello!": {
        "content": "Hi there!",
        "metadata": {"type": "greeting"},
    },
    "condition:has_image=true": {
        "content": "I see an image!",
        "metadata": {"type": "vision"},
    },
    "condition:has_tools=true": {
        "content": "Using tools",
        "tool_calls": [
            {
                "name": "test_tool",
                "parameters": {"param": "value"},
            }
        ],
    },
}


def create_test_provider_config(
    *,
    name: str = "Test Provider",
    model: str = "test/model",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    responses: dict[str, Any] | None = None,
    capabilities: dict[str, Any] | None = None,
    error_rate: float = 0.0,
    delay: float = 0.1,
) -> LLMProviderConfig:
    """Create a standardized test provider configuration.

    Args:
        name: Display name for the test provider
        model: Model identifier for testing
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 4096)
        responses: Mapping of inputs to expected outputs for testing
        capabilities: Provider capability flags (vision, tools, etc.)
        error_rate: Probability of simulated errors (0-1)
        delay: Simulated response delay in seconds

    Returns:
        LLMProviderConfig configured for testing

    Example:
        config = create_test_provider_config(
            name="Vision Test Provider",
            capabilities={"supports_vision": True},
            responses={
                "describe_image": {"content": "I see a cat"},
            }
        )
    """
    return LLMProviderConfig(
        name=name,
        model=model,
        provider="dummy",  # Always use our dummy provider for tests
        temperature=temperature,
        max_tokens=max_tokens,
        metadata={
            "responses": responses or DEFAULT_TEST_RESPONSES,
            "capabilities": capabilities
            or {
                "supports_vision": False,
                "supports_system_messages": True,
                "max_tokens": max_tokens,
            },
            "error_rate": error_rate,
            "delay": delay,
        },
    )


@pytest.fixture
def test_config(
    *,
    provider_name: str = "test-provider",
    context_name: str = "test-context",
    template_name: str = "test-template",
    provider_config: LLMProviderConfig | None = None,
) -> Config:
    """Create a test configuration using DummyProvider."""
    return Config(
        version="1.0",
        global_settings=GlobalSettings(),
        llm_providers={
            provider_name: provider_config or create_test_provider_config(),
        },
        contexts={
            context_name: TextContext(
                content="test content",
                description="Test context",
            ),
        },
        task_templates={
            template_name: TaskTemplate(
                provider=provider_name,
                context=context_name,
            ),
        },
    )


# Specialized test configurations as fixtures
@pytest.fixture
def vision_config() -> Config:
    """Get config for vision testing."""
    return test_config(
        provider_config=create_test_provider_config(
            capabilities={"supports_vision": True},
            responses={
                "condition:has_image=true": {
                    "content": "I see an image of...",
                    "metadata": {"type": "vision"},
                },
            },
        )
    )


@pytest.fixture
def tool_config() -> Config:
    """Get config for tool testing."""
    return test_config(
        provider_config=create_test_provider_config(
            capabilities={"supports_function_calling": True},
            responses={
                "condition:has_tools=true": {
                    "content": "Using tools",
                    "tool_calls": [
                        {
                            "name": "test_tool",
                            "parameters": {"param": "value"},
                        }
                    ],
                },
            },
        )
    )


# Client fixtures using our test configs
@pytest.fixture
async def test_client(test_config):
    """Get configured test client."""
    client = LLMLingClient("test_config.yml")
    client.config_manager = ConfigManager(test_config)
    try:
        await client.startup()
        yield client
    finally:
        await client.shutdown()


@pytest.fixture
async def vision_client(vision_config):
    """Get client configured for vision testing."""
    client = LLMLingClient("test_config.yml")
    client.config_manager = ConfigManager(vision_config)
    try:
        await client.startup()
        yield client
    finally:
        await client.shutdown()


@pytest.fixture
def config_manager(test_config):
    """Get config manager with test configuration."""
    return ConfigManager(test_config)


@pytest.fixture
def processor_registry():
    """Get clean processor registry."""
    return ProcessorRegistry()


@pytest.fixture
def task_executor(processor_registry, config_manager):
    """Get task executor with test configuration."""
    return TaskExecutor(
        context_registry=context_registry,
        processor_registry=processor_registry,
        provider_registry=llm_registry,
        config_manager=config_manager,
    )


@pytest.fixture
def task_manager(test_config, task_executor):
    """Get task manager with test configuration."""
    return TaskManager(test_config, task_executor)
