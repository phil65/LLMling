"""Tests for LLMLing client."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest import mock

import pytest

from llmling.client import ComponentType, LLMLingClient
from llmling.core import exceptions
from llmling.core.exceptions import LLMLingError
from llmling.llm.base import CompletionResult
from llmling.processors.base import ProcessorConfig
from llmling.task.models import TaskResult


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# Test Constants
TEST_CONFIG_PATH = Path("src/llmling/resources/test.yml")
NONEXISTENT_CONFIG_PATH = Path("nonexistent.yml")
TEST_LOG_LEVEL = logging.DEBUG
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
TEST_TEMPLATES = ["quick_review", "detailed_review"]
MAX_CONCURRENT_TASKS = 3

# LLM output related constants
MAX_CONTENT_DIFF_RATIO = 0.5
MIN_CONTENT_LENGTH = 10
MAX_RETRIES = 3

STREAM_TIMEOUT = 30.0
MIN_CHUNKS = 1
MIN_CHUNK_LENGTH = 1
TEST_TEMPLATE = "quick_review"

# Mock response for LLM calls
MOCK_RESPONSE = CompletionResult(
    content="Test response content",
    model="test-model",
    finish_reason="stop",
    metadata={"test": "metadata"},
)


# Common fixtures
@pytest.fixture
def config_path() -> Path:
    """Provide path to test configuration file."""
    if not TEST_CONFIG_PATH.exists():
        msg = f"Test configuration not found: {TEST_CONFIG_PATH}"
        raise FileNotFoundError(msg)
    return TEST_CONFIG_PATH


@pytest.fixture
def components() -> dict[ComponentType, dict[str, Any]]:
    """Provide test components."""
    return {
        "processor": {
            "test_processor": ProcessorConfig(
                type="function",
                import_path="llmling.testing.processors.uppercase_text",
            ),
        },
        "tool": {
            "test_tool": {
                "name": "test_tool",
                "description": "Test tool",
                "import_path": "llmling.testing.tools.example_tool",
            }
        },
    }


@pytest.fixture
def mock_provider():
    """Mock LLM provider with proper async support."""
    with mock.patch("llmling.llm.registry.ProviderRegistry.create_provider") as m:
        provider = mock.AsyncMock()

        async def mock_complete(*args, **kwargs):
            return MOCK_RESPONSE

        provider.complete = mock_complete

        async def mock_stream(*args, **kwargs):
            yield MOCK_RESPONSE

        provider.complete_stream = mock_stream
        provider.model = MOCK_RESPONSE.model
        m.return_value = provider

        # Important: Set up the registry properly
        from llmling.llm.registry import default_registry

        default_registry.reset()
        default_registry["gpt-35-turbo"] = (
            "litellm"  # Map the provider to litellm implementation
        )

        yield provider

        default_registry.reset()


@pytest.fixture
async def async_client(
    config_path: Path,
    components: dict[ComponentType, dict[str, Any]],
    mock_provider,
) -> AsyncGenerator[LLMLingClient, None]:
    """Provide initialized client for async tests."""
    client = LLMLingClient(
        config_path,
        log_level=TEST_LOG_LEVEL,
        components=components,
    )
    await client.startup()
    try:
        yield client
    finally:
        await client.shutdown()


@pytest.mark.unit
class TestClientCreation:
    """Test client initialization and synchronous operations."""

    def test_create_sync(
        self,
        config_path: Path,
        components: dict[ComponentType, dict[str, Any]],
        mock_provider,
    ) -> None:
        """Test synchronous client creation."""
        # Ensure no event loop exists
        asyncio.set_event_loop(None)

        client = LLMLingClient.create(config_path, components=components)
        try:
            assert isinstance(client, LLMLingClient)
            assert client._initialized

            # Test sync execution works
            result = client.execute_sync("quick_review")
            assert isinstance(result, TaskResult)
            assert result.content == MOCK_RESPONSE.content
        finally:
            # Clean up synchronously
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(client.shutdown())
            finally:
                loop.close()

    def test_sync_context_manager(
        self,
        config_path: Path,
        components: dict[ComponentType, dict[str, Any]],
        mock_provider,
    ) -> None:
        """Test synchronous context manager."""
        # Ensure no event loop exists
        asyncio.set_event_loop(None)

        with LLMLingClient.create(config_path, components=components) as client:
            result = client.execute_sync("quick_review")
            assert isinstance(result, TaskResult)
            assert result.content == MOCK_RESPONSE.content

    def test_invalid_config_sync(self) -> None:
        """Test synchronous initialization with invalid configuration."""
        with pytest.raises(exceptions.LLMLingError):
            LLMLingClient.create(NONEXISTENT_CONFIG_PATH)


@pytest.mark.asyncio
@pytest.mark.unit
class TestAsyncOperations:
    """Test asynchronous operations."""

    async def test_async_context_manager(
        self,
        config_path: Path,
        components: dict[ComponentType, dict[str, Any]],
        mock_provider,
    ) -> None:
        """Test async context manager."""
        async with LLMLingClient(config_path, components=components) as client:
            result = await client.execute("quick_review")
            assert isinstance(result, TaskResult)
            assert result.content == MOCK_RESPONSE.content

    async def test_execute_single_task(self, async_client: LLMLingClient) -> None:
        """Test executing a single task."""
        result = await async_client.execute(
            "quick_review",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
        assert result.content == MOCK_RESPONSE.content

    async def test_execute_stream(self, async_client: LLMLingClient) -> None:
        """Test streaming execution."""
        chunks = []
        stream = await async_client.execute(
            "quick_review",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            stream=True,
        )

        assert stream is not None

        async for chunk in stream:
            chunks.append(chunk)
            assert isinstance(chunk, TaskResult)
            assert chunk.content == MOCK_RESPONSE.content
            assert chunk.model == MOCK_RESPONSE.model

        assert len(chunks) >= MIN_CHUNKS

    async def test_concurrent_execution(self, async_client: LLMLingClient) -> None:
        """Test concurrent execution."""
        results = await async_client.execute_many(
            TEST_TEMPLATES,
            max_concurrent=MAX_CONCURRENT_TASKS,
        )
        assert len(results) == len(TEST_TEMPLATES)
        assert all(isinstance(r, TaskResult) for r in results)
        assert all(r.content == MOCK_RESPONSE.content for r in results)


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in both sync and async operations."""

    @pytest.mark.asyncio
    async def test_async_error_handling(self, async_client: LLMLingClient) -> None:
        """Test async error handling."""
        with pytest.raises(LLMLingError):
            await async_client.execute("nonexistent_template")

    def test_sync_error_handling(
        self,
        config_path: Path,
        components: dict[ComponentType, dict[str, Any]],
        mock_provider,
    ) -> None:
        """Test sync error handling."""
        asyncio.set_event_loop(None)
        with (
            LLMLingClient.create(config_path, components=components) as client,
            pytest.raises(LLMLingError),
        ):
            client.execute_sync("nonexistent_template")


@pytest.mark.integration
class TestIntegrationTaskExecution:
    """Integration tests with real LLM."""

    @pytest.fixture
    async def integration_client(
        self,
        config_path: Path,
        components: dict[ComponentType, dict[str, Any]],
    ) -> AsyncGenerator[LLMLingClient, None]:
        """Provide client for integration tests."""
        client = LLMLingClient(
            config_path,
            log_level=TEST_LOG_LEVEL,
            components=components,
        )
        await client.startup()
        try:
            yield client
        finally:
            await client.shutdown()

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_real_llm_execution(self, async_client: LLMLingClient) -> None:
        """Test real LLM execution."""
        result = await async_client.execute(
            "quick_review",
            stream=False,  # Explicit non-streaming
            tools=None,
            tool_choice=None,
        )
        assert result.content
        assert result.model

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_real_llm_streaming(self, async_client: LLMLingClient) -> None:
        """Test real LLM streaming."""
        chunks = [
            chunk
            async for chunk in await async_client.execute(
                "quick_review",
                stream=True,  # Use streaming mode
                tools=None,
                tool_choice=None,
            )
        ]
        assert chunks
        assert all(c.content for c in chunks)

    @staticmethod
    def _validate_task_result(result: TaskResult) -> None:
        """Validate task result structure and content."""
        assert isinstance(result, TaskResult)
        assert result.content
        assert len(result.content) >= MIN_CONTENT_LENGTH
        assert result.model
        assert result.context_metadata
        assert result.completion_metadata

    @staticmethod
    def _validate_chunk(chunk: TaskResult, index: int) -> None:
        """Validate streaming chunk."""
        try:
            assert isinstance(chunk, TaskResult)
            assert chunk.model
            assert isinstance(chunk.content, str)
            assert chunk.context_metadata is not None
            assert chunk.completion_metadata is not None
            assert len(chunk.content) >= MIN_CHUNK_LENGTH

        except AssertionError:
            print(f"\nChunk {index} Validation Error:")
            print(f"Content: {chunk.content[:100]}...")
            print(f"Model: {chunk.model}")
            print(f"Metadata: {chunk.completion_metadata}")
            raise
