"""Tests for LLM components including providers and registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from llmling.core import exceptions
from llmling.llm.base import (
    CompletionResult,
    LLMConfig,
    LLMProvider,
    Message,
)
from llmling.llm.registry import ProviderRegistry


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# Constants for test configuration
TEST_CONTENT = "test content"
TEST_MODEL = "test-model"
TEST_PROVIDER = "test-provider"

# Test messages that any provider should handle
TEST_MESSAGES = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="Hello!"),
]

# Basic config any provider should accept
TEST_CONFIG = LLMConfig(
    model="test/model",
    provider_name="test-provider",
    temperature=0.7,
    max_tokens=100,
    timeout=30,
)

# Extended config with optional features
TEST_CONFIG_FULL = LLMConfig(
    model="test/model",
    provider_name="test-provider",
    temperature=0.7,
    max_tokens=100,
    timeout=30,
    api_base="http://localhost:1234",
    num_retries=3,
    request_timeout=30,
    cache=True,
    metadata={"test": "value"},
)


class MockProvider(LLMProvider):
    """Base mock provider for testing provider interface."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self.calls: list[tuple[list[Message], dict[str, Any]]] = []

    async def complete(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> CompletionResult:
        """Record calls and return test response."""
        self.calls.append((messages, kwargs))
        return CompletionResult(
            content="test response",
            model=self.config.model,
            metadata={"test": "metadata"},
        )

    async def complete_stream(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[CompletionResult]:
        """Stream test chunks."""
        self.calls.append((messages, kwargs))
        yield CompletionResult(
            content="test chunk",
            model=self.config.model,
            metadata={"test": "metadata", "chunk": True},
        )


class FailingProvider(LLMProvider):
    """Provider that simulates failures."""

    async def complete(self, *args: Any, **kwargs: Any) -> CompletionResult:
        msg = "Test failure"
        raise exceptions.LLMError(msg)

    async def complete_stream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[CompletionResult]:
        msg = "Test failure"
        raise exceptions.LLMError(msg)
        yield  # Required for type checking


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
        registry["test"] = MockProvider
        provider = registry.create_provider("test", TEST_CONFIG)
        assert isinstance(provider, MockProvider)

    def test_register_duplicate(self, registry: ProviderRegistry) -> None:
        """Test registering same provider twice."""
        registry["test"] = MockProvider
        with pytest.raises(exceptions.LLMError):
            registry["test"] = MockProvider

    def test_create_unregistered(self, registry: ProviderRegistry) -> None:
        """Test creating unregistered provider."""
        with pytest.raises(exceptions.LLMError):
            registry.create_provider("nonexistent", TEST_CONFIG)


class TestProviderInterface:
    """Tests for the LLM provider interface."""

    @pytest.mark.asyncio
    async def test_provider_settings(self) -> None:
        """Test that provider correctly uses configured settings."""
        provider = MockProvider(TEST_CONFIG)
        result = await provider.complete(TEST_MESSAGES)
        assert result.content == "test response"
        assert result.model == TEST_CONFIG.model
        assert "test" in result.metadata

    @pytest.mark.asyncio
    async def test_provider_streaming(self) -> None:
        """Test provider streaming interface."""
        provider = MockProvider(TEST_CONFIG)
        chunks: list[CompletionResult] = []
        async for chunk in provider.complete_stream(TEST_MESSAGES):
            chunks.append(chunk)
            assert chunk.content == "test chunk"
            assert chunk.model == TEST_CONFIG.model
            assert chunk.metadata.get("chunk") is True

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_provider_error_handling(self) -> None:
        """Test error handling in provider."""
        provider = FailingProvider(TEST_CONFIG)
        with pytest.raises(exceptions.LLMError):
            await provider.complete(TEST_MESSAGES)

        with pytest.raises(exceptions.LLMError):  # noqa: PT012
            async for _ in provider.complete_stream(TEST_MESSAGES):
                pass

    @pytest.mark.asyncio
    async def test_provider_kwargs_handling(self) -> None:
        """Test provider handles additional kwargs correctly."""
        provider = MockProvider(TEST_CONFIG)
        test_kwargs = {"temperature": 0.8, "max_tokens": 200}

        result = await provider.complete(TEST_MESSAGES, **test_kwargs)
        assert result.content == "test response"

        # Verify kwargs were passed through
        assert provider.calls[-1][1] == test_kwargs


if __name__ == "__main__":
    pytest.main(["-vv"])
