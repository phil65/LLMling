from __future__ import annotations

from unittest import mock

import pytest

from llmling.config.models import (
    Config,
    GlobalSettings,
    LLMProviderConfig,
    TextContext,
    ToolConfig,
)
from llmling.context.models import LoadedContext
from llmling.llm.base import CompletionResult
from llmling.task.executor import TaskExecutor
from llmling.task.models import TaskContext, TaskProvider
from llmling.tools.base import ToolRegistry


ANALYZE_IMPORT = "llmling.testing.tools.analyze_ast"
EXAMPLE_IMPORT = "llmling.testing.tools.example_tool"


@pytest.fixture
def tool_config() -> Config:
    """Create a test configuration with tools."""
    return Config(
        version="1.0",
        global_settings=GlobalSettings(timeout=30, max_retries=3, temperature=0.7),
        context_processors={},
        llm_providers={
            "test_provider": LLMProviderConfig(name="ABC", model="test/model"),
        },
        contexts={
            "test_ctx": TextContext(
                type="text", content="Test content", description="Test context"
            ),
        },
        task_templates={},
        tools={
            "analyze": ToolConfig(import_path=ANALYZE_IMPORT, description="Analyze code"),
            "repeat": ToolConfig(import_path=EXAMPLE_IMPORT, name="repeat_text"),
        },
    )


@pytest.fixture
def mock_context_registry() -> mock.MagicMock:
    """Create mock context registry with async support."""
    registry = mock.MagicMock()
    loader = mock.MagicMock()
    context = LoadedContext(content="Test content", source_type="test", metadata={})
    # Make load method a coroutine
    loader.load = mock.AsyncMock(return_value=context)

    registry.get_loader.return_value = loader
    return registry


@pytest.fixture
def mock_processor_registry() -> mock.MagicMock:
    """Create mock processor registry."""
    registry = mock.MagicMock()
    registry.process = mock.AsyncMock()
    return registry


@pytest.fixture
def mock_provider_registry() -> mock.MagicMock:
    """Create mock provider registry with streaming support."""
    registry = mock.MagicMock()

    # Create mock provider with both regular and streaming methods
    mock_provider = mock.MagicMock()

    # Regular completion
    result = CompletionResult(content="Test response", model="test/model", metadata={})
    mock_provider.complete = mock.AsyncMock(return_value=result)

    # Streaming completion will be set by the test that needs it
    mock_provider.complete_stream = None

    registry.create_provider.return_value = mock_provider
    return registry


@pytest.mark.asyncio
async def test_task_with_tools(
    tool_config: Config,
    mock_context_registry: mock.MagicMock,
    mock_processor_registry: mock.MagicMock,
    mock_provider_registry: mock.MagicMock,
) -> None:
    """Test task execution with tools."""
    # Setup
    tool_registry = ToolRegistry()

    # Register tools from config
    for tool_id, tool_cfg in tool_config.tools.items():
        tool_registry.register_path(
            import_path=tool_cfg.import_path,
            name=tool_cfg.name or tool_id,
            description=tool_cfg.description,
        )

    executor = TaskExecutor(
        context_registry=mock_context_registry,
        processor_registry=mock_processor_registry,
        provider_registry=mock_provider_registry,
        tool_registry=tool_registry,
    )

    # Create test context
    test_ctx = tool_config.contexts["test_ctx"]

    # Create and execute task
    tools = ["analyze", "repeat_text"]
    task_ctx = TaskContext(
        context=test_ctx, processors=[], tools=tools, tool_choice="auto"
    )

    task_provider = TaskProvider(
        name="test_provider",
        model="test/model",
        display_name="ABC",
    )

    result = await executor.execute(task_ctx, task_provider)

    # Verify the result
    assert result.content == "Test response"
    assert result.model == "test/model"

    # Verify interactions
    mock_context_registry.get_loader.assert_called_once()
    mock_provider_registry.create_provider.assert_called_once()

    # Verify provider was called with correct configuration
    mock_provider = mock_provider_registry.create_provider.return_value
    mock_provider.complete.assert_called_once()

    # Verify tool configuration was passed correctly
    call_kwargs = mock_provider.complete.call_args[1]
    assert "tools" in call_kwargs
    assert len(call_kwargs["tools"]) == 2  # noqa: PLR2004
    assert call_kwargs["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_task_with_tools_streaming(
    tool_config: Config,
    mock_context_registry: mock.MagicMock,
    mock_processor_registry: mock.MagicMock,
    mock_provider_registry: mock.MagicMock,
) -> None:
    """Test streaming task execution with tools."""

    # Create a proper async generator for streaming
    async def mock_stream(*args, **kwargs):
        for content in ["Chunk 1", "Chunk 2"]:
            yield CompletionResult(content=content, model="test/model", metadata={})

    # Set up mock provider with streaming support
    mock_provider = mock.MagicMock()
    mock_provider.complete_stream = mock_stream
    mock_provider_registry.create_provider.return_value = mock_provider

    # Setup
    tool_registry = ToolRegistry()
    for tool_id, tool_cfg in tool_config.tools.items():
        tool_registry.register_path(
            import_path=tool_cfg.import_path,
            name=tool_cfg.name or tool_id,
            description=tool_cfg.description,
        )

    executor = TaskExecutor(
        context_registry=mock_context_registry,
        processor_registry=mock_processor_registry,
        provider_registry=mock_provider_registry,
        tool_registry=tool_registry,
    )

    test_ctx = tool_config.contexts["test_ctx"]
    tools = ["analyze", "repeat_text"]
    task_ctx = TaskContext(
        context=test_ctx,
        processors=[],
        tools=tools,
        tool_choice="auto",
    )

    task_provider = TaskProvider(
        name="test_provider",
        model="test/model",
        display_name="ABC",
    )
    # Collect streaming results
    chunks = [chunk async for chunk in executor.execute_stream(task_ctx, task_provider)]

    # Verify results
    assert len(chunks) == 2  # noqa: PLR2004
    assert chunks[0].content == "Chunk 1"
    assert chunks[1].content == "Chunk 2"
    assert all(chunk.model == "test/model" for chunk in chunks)

    # Verify interactions
    mock_context_registry.get_loader.assert_called_once()
    mock_provider_registry.create_provider.assert_called_once()

    # Verify provider configuration
    provider_config = mock_provider_registry.create_provider.call_args[0][1]
    assert provider_config.streaming is True  # Verify streaming was enabled
    assert provider_config.model == "test/model"
