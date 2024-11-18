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
MAX_CONTENT_DIFF_RATIO = 0.5  # Allow 50% difference between streaming and non-streaming
MIN_CONTENT_LENGTH = 10  # Minimum expected content length
MAX_RETRIES = 3  # Maximum number of retries for consistency test

STREAM_TIMEOUT = 30.0  # Maximum time to wait for streaming
MIN_CHUNKS = 1  # Minimum number of chunks expected
MIN_CHUNK_LENGTH = 1  # Minimum length of each chunk
TEST_TEMPLATE = "quick_review"  # Template known to work

# Mock response for LLM calls
MOCK_RESPONSE = CompletionResult(
    content="Test response content",
    model="test-model",
    finish_reason="stop",
    metadata={"test": "metadata"},
)


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
    }


@pytest.fixture
def mock_llm_response() -> CompletionResult:
    """Provide mock LLM response."""
    return MOCK_RESPONSE


@pytest.fixture
def mock_provider():
    """Mock LLM provider."""
    with mock.patch("llmling.llm.registry.ProviderRegistry.create_provider") as m:
        provider = mock.AsyncMock()
        provider.complete.return_value = MOCK_RESPONSE

        # Properly mock the streaming response
        async def mock_stream(*args, **kwargs):
            yield MOCK_RESPONSE

        provider.complete_stream = mock_stream
        m.return_value = provider
        yield provider


@pytest.fixture
async def client(
    config_path: Path,
    components: dict[ComponentType, dict[str, Any]],
    mock_provider,
) -> AsyncGenerator[LLMLingClient, None]:
    """Provide initialized LLMLing client."""
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
    """Test client initialization and context managers."""

    def test_create_sync(
        self,
        config_path: Path,
        components: dict[ComponentType, dict[str, Any]],
        mock_provider,
    ) -> None:
        """Test synchronous client creation."""
        client = LLMLingClient.create(config_path, components=components)
        assert isinstance(client, LLMLingClient)
        assert client._initialized

    def test_sync_context_manager(
        self,
        config_path: Path,
        components: dict[ComponentType, dict[str, Any]],
        mock_provider,
    ) -> None:
        """Test synchronous context manager."""
        with LLMLingClient.create(config_path, components=components) as client:
            result = client.execute_sync("quick_review")
            assert isinstance(result, TaskResult)
            assert result.content == MOCK_RESPONSE.content

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_client_invalid_config(self) -> None:
        """Test client initialization with invalid configuration."""
        client = LLMLingClient(NONEXISTENT_CONFIG_PATH)
        with pytest.raises(exceptions.LLMLingError):
            await client.startup()


@pytest.mark.unit
class TestMockedTaskExecution:
    """Unit tests with mocked LLM responses."""

    @pytest.mark.asyncio
    async def test_execute_single_task(self, client: LLMLingClient) -> None:
        """Test executing a single task with mocked LLM."""
        result = await client.execute(
            "quick_review",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
        assert result.content == MOCK_RESPONSE.content

    @pytest.mark.asyncio
    async def test_execute_stream(self, client: LLMLingClient) -> None:
        """Test streaming execution with mocked LLM."""
        chunks = []
        stream = await client.execute(
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

    @pytest.mark.asyncio
    async def test_concurrent_execution(self, client: LLMLingClient) -> None:
        """Test concurrent task execution."""
        results = await client.execute_many(
            TEST_TEMPLATES,
            max_concurrent=MAX_CONCURRENT_TASKS,
        )
        assert len(results) == len(TEST_TEMPLATES)
        assert all(isinstance(r, TaskResult) for r in results)
        assert all(r.content == MOCK_RESPONSE.content for r in results)

    @pytest.mark.asyncio
    async def test_error_handling(self, client: LLMLingClient) -> None:
        """Test error handling for invalid templates."""
        with pytest.raises(LLMLingError):
            await client.execute("nonexistent_template")


@pytest.mark.integration
class TestIntegrationTaskExecution:
    """Integration tests for task execution."""

    @pytest.fixture
    async def integration_client(
        self,
        config_path: Path,
        components: dict[ComponentType, dict[str, Any]],
    ) -> AsyncGenerator[LLMLingClient, None]:
        """Provide client for integration tests without mocks."""
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
    async def test_real_llm_execution(self, integration_client: LLMLingClient) -> None:
        """Test executing a task with real LLM."""
        result = await integration_client.execute(
            "quick_review",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
        self._validate_task_result(result)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_real_llm_streaming(self, integration_client: LLMLingClient) -> None:
        """Test streaming with real LLM."""
        chunks = []
        total_content = ""

        try:
            async with asyncio.timeout(STREAM_TIMEOUT):
                async for chunk in await integration_client.execute(
                    TEST_TEMPLATE,
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    stream=True,
                ):
                    chunks.append(chunk)
                    total_content += chunk.content
                    self._validate_chunk(chunk, len(chunks) - 1)

        except TimeoutError as exc:
            msg = f"Streaming timed out after {STREAM_TIMEOUT}s"
            raise AssertionError(msg) from exc

        assert len(chunks) >= MIN_CHUNKS
        assert len(total_content) >= MIN_CONTENT_LENGTH

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
        """Validate individual stream chunk."""
        try:
            assert isinstance(chunk, TaskResult), (
                f"Chunk {index}: Invalid type {type(chunk)}"
            )
            assert chunk.model, f"Chunk {index}: Missing model"
            assert isinstance(chunk.content, str), f"Chunk {index}: Content is not string"
            assert chunk.context_metadata is not None, (
                f"Chunk {index}: Missing context metadata"
            )
            assert chunk.completion_metadata is not None, (
                f"Chunk {index}: Missing completion metadata"
            )
            assert len(chunk.content) >= MIN_CHUNK_LENGTH, (
                f"Chunk {index}: Content too short ({len(chunk.content)} chars)"
            )

        except AssertionError:
            print(f"\nChunk {index} Validation Error:")
            print(f"Content: {chunk.content[:100]}...")
            print(f"Model: {chunk.model}")
            print(f"Metadata: {chunk.completion_metadata}")
            raise


@pytest.mark.unit
class TestCustomization:
    """Test client customization options."""

    @pytest.mark.asyncio
    async def test_custom_components(
        self,
        config_path: Path,
        components: dict[ComponentType, dict[str, Any]],
        mock_provider,
    ) -> None:
        """Test execution with custom components."""
        client = LLMLingClient(config_path, components=components)
        await client.startup()
        try:
            result = await client.execute("quick_review")
            assert isinstance(result, TaskResult)
            assert result.content == MOCK_RESPONSE.content
        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_component_registration(
        self,
        config_path: Path,
        components: dict[ComponentType, dict[str, Any]],
        mock_provider,
    ) -> None:
        """Test that components are properly registered."""
        client = LLMLingClient(config_path, components=components)
        await client.startup()

        try:
            # Verify processor registration
            if "processor" in components:
                for name in components["processor"]:
                    assert client.processor_registry
                    assert name in client.processor_registry._processors

            # Verify successful task execution with custom components
            result = await client.execute("quick_review")
            assert isinstance(result, TaskResult)
            assert result.content

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_invalid_component_type(self, config_path: Path) -> None:
        """Test handling of invalid component types."""
        invalid_components = {
            "invalid_type": {"test": "value"}  # type: ignore
        }

        client = LLMLingClient(config_path, components=invalid_components)  # type: ignore
        await client.startup()

        try:
            # Should still work even with invalid component type
            result = await client.execute("quick_review")
            assert isinstance(result, TaskResult)
        finally:
            await client.shutdown()

    def test_sync_execution(self, config_path: Path) -> None:
        """Test synchronous execution methods."""
        with LLMLingClient.create(config_path) as client:
            # Test single execution
            result = client.execute_sync("quick_review")
            assert isinstance(result, TaskResult)
            assert result.content

            # Test concurrent execution
            results = client.execute_many_sync(TEST_TEMPLATES)
            assert len(results) == len(TEST_TEMPLATES)
            assert all(isinstance(r, TaskResult) for r in results)
