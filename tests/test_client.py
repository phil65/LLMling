from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from llmling.client import LLMLingClient
from llmling.core.exceptions import LLMLingError
from llmling.processors.base import ProcessorConfig
from llmling.task.models import TaskResult


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# Constants
TEST_CONFIG_PATH = Path("src/llmling/resources/test.yml")
NONEXISTENT_CONFIG_PATH = Path("nonexistent.yml")
TEST_LOG_LEVEL = logging.DEBUG
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
TEST_TEMPLATES = ["quick_review", "detailed_review"]
MAX_CONCURRENT_TASKS = 3
STREAM_CHUNK_SIZE = 50


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


def test_create_sync(config_path: Path) -> None:
    """Test synchronous client creation."""
    client = LLMLingClient.create(config_path)
    assert isinstance(client, LLMLingClient)
    assert client._initialized


def test_sync_context_manager(config_path: Path) -> None:
    """Test synchronous context manager."""
    with LLMLingClient.create(config_path) as client:
        result = client.execute_sync("quick_review")
        assert isinstance(result, TaskResult)
        assert result.content


@pytest.mark.asyncio
async def test_async_context_manager(config_path: Path) -> None:
    """Test async context manager."""
    async with LLMLingClient(config_path) as client:
        result = await client.execute("quick_review")
        assert isinstance(result, TaskResult)
        assert result.content


@pytest.mark.asyncio
async def test_client_invalid_config() -> None:
    """Test client initialization with invalid configuration."""
    with pytest.raises((FileNotFoundError, LLMLingError)):
        client = LLMLingClient(NONEXISTENT_CONFIG_PATH)
        await client.startup()


@pytest.mark.asyncio
async def test_execute_single_task(client: LLMLingClient) -> None:
    """Test executing a single task."""
    result = await client.execute("quick_review", system_prompt=DEFAULT_SYSTEM_PROMPT)
    assert isinstance(result, TaskResult)
    assert result.content
    assert result.model
    assert result.context_metadata
    assert result.completion_metadata


@pytest.mark.asyncio
async def test_execute_stream(client: LLMLingClient) -> None:
    """Test streaming execution."""
    chunks = []
    stream = await client.execute(
        "quick_review", system_prompt=DEFAULT_SYSTEM_PROMPT, stream=True
    )
    async for chunk in stream:  # Only use async for on the stream result
        assert isinstance(chunk, TaskResult)
        assert chunk.content
        chunks.append(chunk)
    assert chunks
    assert all(len(c.content) > 0 for c in chunks)


@pytest.mark.asyncio
async def test_concurrent_execution(client: LLMLingClient) -> None:
    """Test concurrent task execution."""
    results = await client.execute_many(
        TEST_TEMPLATES, max_concurrent=MAX_CONCURRENT_TASKS
    )
    assert len(results) == len(TEST_TEMPLATES)
    assert all(isinstance(r, TaskResult) for r in results)
    assert all(r.content for r in results)


@pytest.mark.asyncio
async def test_stream_consistency(client: LLMLingClient) -> None:
    """Test that streaming results are consistent with non-streaming."""
    # Get non-streaming result
    full_result = await client.execute("quick_review")

    # Get streaming result
    streamed_content = ""
    stream = await client.execute("quick_review", stream=True)
    async for chunk in stream:  # Only use async for on the stream result
        streamed_content += chunk.content

    # Content should be similar in length
    assert abs(len(streamed_content) - len(full_result.content)) < STREAM_CHUNK_SIZE


@pytest.mark.asyncio
async def test_error_handling(client: LLMLingClient) -> None:
    """Test error handling for invalid templates."""
    with pytest.raises(LLMLingError):
        await client.execute("nonexistent_template")


@pytest.mark.asyncio
async def test_custom_processors(
    config_path: Path, custom_processors: dict[str, ProcessorConfig]
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


def test_sync_execution(config_path: Path) -> None:
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
