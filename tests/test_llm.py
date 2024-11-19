"""Tests for LLM components including providers and registry."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest import mock

import pytest

from llmling.core import exceptions
from llmling.llm.base import (
    CompletionResult,
    LLMConfig,
    Message,
    RetryableProvider,
)
from llmling.llm.providers.litellm import LiteLLMProvider
from llmling.llm.registry import ProviderRegistry


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# Test data
TEST_MESSAGES = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="Hello!"),
]

TEST_CONFIG = LLMConfig(
    model="ollama/llama3.2:3b",
    provider_name="test-provider",
    temperature=0.7,
    max_tokens=100,
    timeout=5,
    max_retries=2,
)

MOCK_COMPLETION = CompletionResult(
    content="Test response",
    model="test/model",
    finish_reason="stop",
    metadata={"provider": "test"},
)

STREAM_CHUNKS = [
    CompletionResult(
        content="Test ",
        model="test/model",
        is_stream_chunk=True,
        metadata={"provider": "test"},
    ),
    CompletionResult(
        content="response",
        model="test/model",
        is_stream_chunk=True,
        metadata={"provider": "test"},
    ),
]


class MockProvider(RetryableProvider):
    """Mock provider for testing retry logic."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        fail_times: int = 0,
        delay: float = 0.1,
    ) -> None:
        """Initialize with number of times to fail before succeeding."""
        super().__init__(config)
        self.fail_times = fail_times
        self.attempts = 0
        self.delay = delay

    async def _complete_impl(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> CompletionResult:
        """Implement completion with configurable failures."""
        self.attempts += 1
        await asyncio.sleep(self.delay)  # Simulate work

        if self.attempts <= self.fail_times:
            msg = f"Mock failure {self.attempts}"
            raise ValueError(msg)

        return MOCK_COMPLETION

    async def _complete_stream_impl(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[CompletionResult]:
        """Implement streaming with configurable failures."""
        self.attempts += 1
        await asyncio.sleep(self.delay)

        if self.attempts <= self.fail_times:
            msg = f"Mock stream failure {self.attempts}"
            raise ValueError(msg)

        for chunk in STREAM_CHUNKS:
            yield chunk


@pytest.fixture
def registry() -> ProviderRegistry:
    """Create a fresh provider registry."""
    registry = ProviderRegistry()
    registry.reset()
    return registry


class TestProviderRegistry:
    """Tests for the provider registry."""

    def test_register_provider(self, registry: ProviderRegistry) -> None:
        """Test basic provider registration."""
        registry.register_provider("test", "litellm")
        provider = registry.create_provider("test", TEST_CONFIG)
        assert isinstance(provider, LiteLLMProvider)

    def test_register_duplicate(self, registry: ProviderRegistry) -> None:
        """Test registering same provider twice."""
        registry.register_provider("test", "litellm")
        # Same implementation should be fine
        registry.register_provider("test", "litellm")

        # Different implementation should raise
        with pytest.raises(exceptions.LLMError):
            registry.register_provider("test", "different")

    def test_create_unregistered(self, registry: ProviderRegistry) -> None:
        """Test creating unregistered provider."""
        with pytest.raises(exceptions.LLMError):
            registry.create_provider("nonexistent", TEST_CONFIG)


@pytest.mark.slow
class TestRetryableProvider:
    """Tests for the retryable provider base class."""

    @pytest.mark.asyncio
    async def test_retry_success(self) -> None:
        """Test successful retry after failures."""
        provider = MockProvider(TEST_CONFIG, fail_times=1)
        result = await provider.complete(TEST_MESSAGES)
        assert result == MOCK_COMPLETION
        # One failure + success = 2
        assert provider.attempts == 2  # noqa: PLR2004

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self) -> None:
        """Test when all retries are exhausted."""
        provider = MockProvider(
            TEST_CONFIG,
            fail_times=TEST_CONFIG.max_retries + 1,
        )
        with pytest.raises(exceptions.LLMError):
            await provider.complete(TEST_MESSAGES)
        assert provider.attempts == TEST_CONFIG.max_retries + 1

    @pytest.mark.asyncio
    async def test_streaming_retry(self) -> None:
        """Test retry logic in streaming mode."""
        provider = MockProvider(TEST_CONFIG, fail_times=1)
        chunks = [chunk async for chunk in provider.complete_stream(TEST_MESSAGES)]
        assert chunks == STREAM_CHUNKS
        # One failure + success = 2
        assert provider.attempts == 2  # noqa: PLR2004


class TestLiteLLMProvider:
    """Tests for the LiteLLM provider implementation."""

    @pytest.mark.asyncio
    async def test_completion(self) -> None:
        """Test basic completion."""
        with mock.patch("litellm.acompletion") as mock_complete:
            mock_complete.return_value = mock.MagicMock(
                choices=[
                    mock.MagicMock(
                        message=mock.MagicMock(content="Test response"),
                        finish_reason="stop",
                    )
                ],
                model="test/model",
                usage=mock.MagicMock(
                    model_dump=mock.MagicMock(return_value={"total_tokens": 10})
                ),
            )

            provider = LiteLLMProvider(TEST_CONFIG)
            result = await provider.complete(TEST_MESSAGES)

            assert result.content == "Test response"
            assert result.model == "test/model"
            assert result.metadata["provider"] == "litellm"

    @pytest.mark.asyncio
    async def test_streaming(self) -> None:
        """Test streaming completion."""
        mock_chunks = [
            mock.MagicMock(
                choices=[
                    mock.MagicMock(
                        delta=mock.MagicMock(content="Test "),
                        finish_reason=None,
                    )
                ],
                model="test/model",
            ),
            mock.MagicMock(
                choices=[
                    mock.MagicMock(
                        delta=mock.MagicMock(content="response"),
                        finish_reason="stop",
                    )
                ],
                model="test/model",
            ),
        ]

        async def mock_stream() -> AsyncIterator[Any]:
            for chunk in mock_chunks:
                yield chunk

        with mock.patch("litellm.acompletion") as mock_complete:
            mock_complete.return_value = mock_stream()

            provider = LiteLLMProvider(TEST_CONFIG)
            chunks = [chunk async for chunk in provider.complete_stream(TEST_MESSAGES)]

            assert len(chunks) == 2  # noqa: PLR2004
            assert chunks[0].content == "Test "
            assert chunks[1].content == "response"
            assert all(c.model == "test/model" for c in chunks)
            assert all(c.metadata["provider"] == "litellm" for c in chunks)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_error_handling(self) -> None:
        """Test error handling."""
        with mock.patch("litellm.acompletion", side_effect=ValueError("API error")):
            provider = LiteLLMProvider(TEST_CONFIG)
            with pytest.raises(exceptions.LLMError):
                await provider.complete(TEST_MESSAGES)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
