from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from llmling.client import LLMLingClient
from llmling.core import exceptions
from llmling.core.exceptions import LLMLingError
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


@pytest.fixture
def config_path() -> Path:
    """Provide path to test configuration file."""
    if not TEST_CONFIG_PATH.exists():
        msg = f"Test configuration not found: {TEST_CONFIG_PATH}"
        raise FileNotFoundError(msg)
    return TEST_CONFIG_PATH


@pytest.fixture
async def client(config_path: Path) -> AsyncGenerator[LLMLingClient, None]:
    """Provide initialized LLMLing client."""
    client = LLMLingClient(config_path, log_level=TEST_LOG_LEVEL)
    await client.startup()
    try:
        yield client
    finally:
        await client.shutdown()


@pytest.fixture
def custom_processors() -> dict[str, ProcessorConfig]:
    """Provide test processor configurations."""
    return {
        "test_processor": ProcessorConfig(
            type="function",
            import_path="llmling.testing.processors.uppercase_text",
        )
    }


class TestClientCreation:
    """Test client initialization and context managers."""

    def test_create_sync(self, config_path: Path) -> None:
        """Test synchronous client creation."""
        client = LLMLingClient.create(config_path)
        assert isinstance(client, LLMLingClient)
        assert client._initialized

    def test_sync_context_manager(self, config_path: Path) -> None:
        """Test synchronous context manager."""
        with LLMLingClient.create(config_path) as client:
            result = client.execute_sync("quick_review")
            assert isinstance(result, TaskResult)
            assert result.content

    @pytest.mark.asyncio
    async def test_async_context_manager(self, config_path: Path) -> None:
        """Test async context manager."""
        async with LLMLingClient(config_path) as client:
            result = await client.execute("quick_review")
            assert isinstance(result, TaskResult)
            assert result.content

    @pytest.mark.asyncio
    async def test_client_invalid_config(self) -> None:
        """Test client initialization with invalid configuration."""
        with pytest.raises(exceptions.LLMLingError):
            client = LLMLingClient(NONEXISTENT_CONFIG_PATH)
            await client.startup()


class TestTaskExecution:
    """Test task execution functionality."""

    @pytest.mark.asyncio
    async def test_execute_single_task(self, client: LLMLingClient) -> None:
        """Test executing a single task."""
        result = await client.execute("quick_review", system_prompt=DEFAULT_SYSTEM_PROMPT)
        self._validate_task_result(result)

    @pytest.mark.asyncio
    async def test_execute_stream(self, client: LLMLingClient) -> None:
        """Test streaming execution."""
        chunks: list[TaskResult] = []
        total_content = ""
        error: Exception | None = None

        try:
            # First verify the template exists and works in non-streaming mode
            normal_result = await client.execute(
                TEST_TEMPLATE, system_prompt=DEFAULT_SYSTEM_PROMPT, stream=False
            )
            assert normal_result.content, "Non-streaming execution failed"

            # Now test streaming
            stream_iterator = await client.execute(
                TEST_TEMPLATE, system_prompt=DEFAULT_SYSTEM_PROMPT, stream=True
            )

            assert stream_iterator is not None, "Stream iterator is None"

            # Collect chunks with timeout and debug logging
            async with asyncio.timeout(STREAM_TIMEOUT):
                async for chunk in stream_iterator:
                    # Debug logging
                    print(f"Received chunk: {len(chunk.content)} chars")

                    chunks.append(chunk)
                    total_content += chunk.content

                    # Validate chunk immediately
                    self._validate_chunk(chunk, len(chunks) - 1)

            print(f"Total chunks received: {len(chunks)}")
            print(f"Total content length: {len(total_content)}")

        except TimeoutError:
            error = AssertionError(f"Streaming timed out after {STREAM_TIMEOUT}s")
        except Exception as exc:
            error = exc
            print(f"Error during streaming: {exc}")

        # If we had an error, provide diagnostic information
        if error:
            print("\nDiagnostic Information:")
            print(f"Chunks received: {len(chunks)}")
            print(f"Total content length: {len(total_content)}")
            if chunks:
                print("\nFirst chunk content:")
                print(chunks[0].content[:100] + "...")
            raise error

        # Final validations
        assert len(chunks) >= MIN_CHUNKS, (
            f"Expected at least {MIN_CHUNKS} chunks, got {len(chunks)}"
        )
        assert len(total_content) >= MIN_CONTENT_LENGTH, (
            f"Total content too short: {len(total_content)} chars"
        )

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

            # Content validation
            content_length = len(chunk.content)
            assert content_length >= MIN_CHUNK_LENGTH, (
                f"Chunk {index}: Content too short ({content_length} chars)"
            )

        except AssertionError:
            print(f"\nChunk {index} Validation Error:")
            print(f"Content: {chunk.content[:100]}...")
            print(f"Model: {chunk.model}")
            print(f"Metadata: {chunk.completion_metadata}")
            raise

    async def test_stream_basic(self, client: LLMLingClient) -> None:
        """Simplified streaming test for debugging."""
        stream = await client.execute(TEST_TEMPLATE, stream=True)

        assert stream is not None, "Stream is None"

        chunk_count = 0
        async for chunk in stream:
            chunk_count += 1
            assert chunk.content, f"Empty content in chunk {chunk_count}"

        assert chunk_count > 0, "No chunks received"

    @pytest.mark.asyncio
    async def test_concurrent_execution(self, client: LLMLingClient) -> None:
        """Test concurrent task execution."""
        results = await client.execute_many(
            TEST_TEMPLATES,
            max_concurrent=MAX_CONCURRENT_TASKS,
        )
        assert len(results) == len(TEST_TEMPLATES)
        assert all(isinstance(r, TaskResult) for r in results)
        for result in results:
            self._validate_task_result(result)

    @staticmethod
    def _validate_task_result(result: TaskResult) -> None:
        """Validate task result structure and content."""
        assert isinstance(result, TaskResult)
        assert result.content
        assert len(result.content) >= MIN_CONTENT_LENGTH
        assert result.model
        assert result.context_metadata
        assert result.completion_metadata

    @pytest.mark.asyncio
    async def test_stream_consistency(self, client: LLMLingClient) -> None:
        """Test that streaming results are reasonably consistent with non-streaming.

        This test allows for some variation in content length between streaming
        and non-streaming modes, as LLMs might produce slightly different outputs.
        It retries a few times if the difference is too large.
        """
        for attempt in range(MAX_RETRIES):
            try:
                # Get non-streaming result
                full_result = await client.execute(
                    "quick_review", system_prompt=DEFAULT_SYSTEM_PROMPT
                )
                full_len = len(full_result.content)

                # Get streaming result
                streamed_content = ""
                async for chunk in await client.execute(
                    "quick_review", system_prompt=DEFAULT_SYSTEM_PROMPT, stream=True
                ):
                    streamed_content += chunk.content
                streamed_len = len(streamed_content)

                # Calculate ratio and difference
                ratio = abs(full_len - streamed_len) / max(full_len, streamed_len)

                # Print diagnostic information
                print(f"\nAttempt {attempt + 1}:")
                print(f"Full content length: {full_len}")
                print(f"Streamed content length: {streamed_len}")
                print(f"Difference ratio: {ratio:.2f}")

                # Check if lengths are within acceptable range
                if ratio <= MAX_CONTENT_DIFF_RATIO:
                    # Success case
                    assert full_len >= MIN_CONTENT_LENGTH, (
                        f"Full content too short: {full_len}"
                    )
                    assert streamed_len >= MIN_CONTENT_LENGTH, (
                        f"Streamed content too short: {streamed_len}"
                    )
                    return  # Test passed

                # If ratio is too large, try again
                print(f"Ratio {ratio:.2f} exceeds maximum {MAX_CONTENT_DIFF_RATIO}")
                await asyncio.sleep(1)  # Brief delay between retries

            except Exception as exc:
                print(f"Error on attempt {attempt + 1}: {exc}")
                if attempt == MAX_RETRIES - 1:
                    raise

        # If we get here, all attempts failed
        msg = (
            f"Content length difference too large after {MAX_RETRIES} attempts. "
            f"Last attempt: full={full_len}, streamed={streamed_len}, "
            f"ratio={ratio:.2f}"
        )
        raise AssertionError(msg)

    @pytest.mark.asyncio
    async def test_error_handling(self, client: LLMLingClient) -> None:
        """Test error handling for invalid templates."""
        with pytest.raises(LLMLingError):
            await client.execute("nonexistent_template")


class TestCustomization:
    """Test client customization options."""

    @pytest.mark.asyncio
    async def test_custom_processors(
        self, config_path: Path, custom_processors: dict[str, ProcessorConfig]
    ) -> None:
        """Test execution with custom processors."""
        client = LLMLingClient(config_path, processors=custom_processors)
        await client.startup()
        try:
            result = await client.execute("quick_review")
            assert isinstance(result, TaskResult)
            assert result.content
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
