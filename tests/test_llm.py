"""Tests for LLM components including providers and registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest import mock

import litellm
from litellm.utils import ModelResponse
import pytest

from llmling.core import exceptions
from llmling.core.exceptions import LLMError
from llmling.llm.base import (
    LLMConfig,
    Message,
)
from llmling.llm.providers.litellm import LiteLLMProvider
from llmling.llm.registry import ProviderRegistry


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator


# Test data
TEST_MESSAGES = cast(
    list[Message],
    [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello!"),
    ],
)

# Basic config without LiteLLM specific settings
TEST_CONFIG = LLMConfig(
    model="openai/gpt-3.5-turbo",
    provider_name="test-provider",
    temperature=0.7,
    max_tokens=100,
    timeout=30,
)

# Config with LiteLLM settings
TEST_CONFIG_FULL = LLMConfig(
    model="openai/gpt-3.5-turbo",
    provider_name="test-provider",
    temperature=0.7,
    max_tokens=100,
    timeout=30,
    api_base="http://localhost:11434",
    num_retries=3,
    request_timeout=30,
    cache=True,
    metadata={"test": "value"},
)


def create_mock_litellm_response(
    content: str = "Test response",
    model: str = "test/model",
    finish_reason: str = "stop",
) -> ModelResponse:
    """Create a properly structured mock LiteLLM response."""
    message_mock = mock.MagicMock()
    message_mock.content = content
    message_mock.tool_calls = None

    choice_mock = mock.MagicMock()
    choice_mock.message = message_mock
    choice_mock.finish_reason = finish_reason

    usage_mock = mock.MagicMock()
    usage_mock.model_dump.return_value = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }

    response_mock = cast(ModelResponse, mock.MagicMock())
    response_mock.choices = [choice_mock]
    response_mock.model = model
    response_mock.usage = usage_mock

    return response_mock


def create_mock_litellm_stream_chunk(
    content: str,
    model: str = "test/model",
    finish_reason: str | None = None,
) -> ModelResponse:
    """Create a properly structured mock LiteLLM stream chunk."""
    delta_mock = mock.MagicMock()
    delta_mock.content = content
    delta_mock.tool_calls = None

    choice_mock = mock.MagicMock()
    choice_mock.delta = delta_mock
    choice_mock.finish_reason = finish_reason

    chunk_mock = cast(ModelResponse, mock.MagicMock())
    chunk_mock.choices = [choice_mock]
    chunk_mock.model = model

    return chunk_mock


@pytest.fixture
def reset_litellm() -> Generator[None, None, None]:  # noqa: PT004
    """Reset LiteLLM global settings before each test."""
    original_api_base = getattr(litellm, "api_base", None)
    original_api_key = getattr(litellm, "api_key", None)

    litellm.api_base = None
    litellm.api_key = None

    yield

    litellm.api_base = original_api_base
    litellm.api_key = original_api_key


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
        registry.register_provider("test", "litellm")

        with pytest.raises(exceptions.LLMError):
            registry.register_provider("test", "different")

    def test_create_unregistered(self, registry: ProviderRegistry) -> None:
        """Test creating unregistered provider."""
        with pytest.raises(exceptions.LLMError):
            registry.create_provider("nonexistent", TEST_CONFIG)


class TestLiteLLMProvider:
    """Tests for the LiteLLM provider implementation."""

    @pytest.mark.asyncio
    async def test_provider_settings(self, reset_litellm: None) -> None:
        """Test that provider correctly uses configured settings."""
        config = LLMConfig(
            model="test/model",
            provider_name="test",
            api_base="https://test.com",
            api_key="test-key",
        )

        mock_response = create_mock_litellm_response()
        mock_acompletion_called_with: dict[str, Any] = {}

        async def mock_acompletion(**kwargs: Any) -> ModelResponse:
            nonlocal mock_acompletion_called_with
            mock_acompletion_called_with = kwargs
            return mock_response

        with mock.patch("litellm.acompletion", new=mock_acompletion):
            provider = LiteLLMProvider(config)
            await provider.complete(TEST_MESSAGES)

            assert mock_acompletion_called_with["api_base"] == "https://test.com"
            assert mock_acompletion_called_with["api_key"] == "test-key"

    @pytest.mark.asyncio
    async def test_config_override_globals(self, reset_litellm: None) -> None:
        """Test that request config overrides global defaults."""
        litellm.api_base = "https://global.test"
        litellm.api_key = "global-key"

        config = LLMConfig(
            model="test/model",
            provider_name="test",
            api_base="https://local.test",
            api_key="local-key",
        )

        mock_response = create_mock_litellm_response()
        mock_acompletion_called_with: dict[str, Any] = {}

        async def mock_acompletion(**kwargs: Any) -> ModelResponse:
            nonlocal mock_acompletion_called_with
            mock_acompletion_called_with = kwargs
            return mock_response

        with mock.patch("litellm.acompletion", new=mock_acompletion):
            provider = LiteLLMProvider(config)
            await provider.complete(TEST_MESSAGES)

            assert mock_acompletion_called_with["api_base"] == "https://local.test"
            assert mock_acompletion_called_with["api_key"] == "local-key"

    @pytest.mark.asyncio
    async def test_kwargs_override_all(self, reset_litellm: None) -> None:
        """Test that kwargs override both globals and config."""
        litellm.api_base = "https://global.test"

        config = LLMConfig(
            model="test/model", provider_name="test", api_base="https://local.test"
        )

        mock_response = create_mock_litellm_response()
        mock_acompletion_called_with: dict[str, Any] = {}

        async def mock_acompletion(**kwargs: Any) -> ModelResponse:
            nonlocal mock_acompletion_called_with
            mock_acompletion_called_with = kwargs
            return mock_response

        with mock.patch("litellm.acompletion", new=mock_acompletion):
            provider = LiteLLMProvider(config)
            await provider.complete(TEST_MESSAGES, api_base="https://override.test")

            assert mock_acompletion_called_with["api_base"] == "https://override.test"

    @pytest.mark.asyncio
    async def test_none_values_not_sent(self, reset_litellm: None) -> None:
        """Test that None values aren't included in request."""
        config = LLMConfig(
            model="test/model",
            provider_name="test",
            api_base=None,
            api_key=None,
        )

        mock_response = create_mock_litellm_response()
        mock_acompletion_called_with: dict[str, Any] = {}

        async def mock_acompletion(**kwargs: Any) -> ModelResponse:
            nonlocal mock_acompletion_called_with
            mock_acompletion_called_with = kwargs
            return mock_response

        with mock.patch("litellm.acompletion", new=mock_acompletion):
            provider = LiteLLMProvider(config)
            await provider.complete(TEST_MESSAGES)

            assert "api_base" not in mock_acompletion_called_with
            assert "api_key" not in mock_acompletion_called_with

    @pytest.mark.asyncio
    async def test_basic_completion(self) -> None:
        """Test basic completion with minimal config."""
        mock_response = create_mock_litellm_response()
        mock_acompletion_called_with: dict[str, Any] = {}

        async def mock_acompletion(**kwargs: Any) -> ModelResponse:
            nonlocal mock_acompletion_called_with
            mock_acompletion_called_with = kwargs
            return mock_response

        with mock.patch("litellm.acompletion", new=mock_acompletion):
            provider = LiteLLMProvider(TEST_CONFIG)
            result = await provider.complete(TEST_MESSAGES)

            assert mock_acompletion_called_with["model"] == TEST_CONFIG.model
            assert mock_acompletion_called_with["temperature"] == TEST_CONFIG.temperature
            assert mock_acompletion_called_with["max_tokens"] == TEST_CONFIG.max_tokens

            assert result.content == "Test response"
            assert result.model == "test/model"
            assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_completion_with_full_config(self) -> None:
        """Test completion with full LiteLLM configuration."""
        mock_response = create_mock_litellm_response()
        mock_acompletion_called_with: dict[str, Any] = {}

        async def mock_acompletion(**kwargs: Any) -> ModelResponse:
            nonlocal mock_acompletion_called_with
            mock_acompletion_called_with = kwargs
            return mock_response

        with mock.patch("litellm.acompletion", new=mock_acompletion):
            provider = LiteLLMProvider(TEST_CONFIG_FULL)
            _ = await provider.complete(TEST_MESSAGES)

            assert mock_acompletion_called_with["model"] == TEST_CONFIG_FULL.model
            assert (
                mock_acompletion_called_with["temperature"]
                == TEST_CONFIG_FULL.temperature
            )
            assert (
                mock_acompletion_called_with["max_tokens"] == TEST_CONFIG_FULL.max_tokens
            )
            assert mock_acompletion_called_with["api_base"] == TEST_CONFIG_FULL.api_base
            assert (
                mock_acompletion_called_with["num_retries"]
                == TEST_CONFIG_FULL.num_retries
            )
            assert (
                mock_acompletion_called_with["request_timeout"]
                == TEST_CONFIG_FULL.request_timeout
            )
            assert mock_acompletion_called_with["cache"] is True
            assert mock_acompletion_called_with["metadata"] == {"test": "value"}

    @pytest.mark.asyncio
    async def test_streaming(self) -> None:
        """Test streaming completion."""
        mock_chunks = [
            create_mock_litellm_stream_chunk(content="Test ", finish_reason=None),
            create_mock_litellm_stream_chunk(content="response", finish_reason="stop"),
        ]

        async def mock_stream(**kwargs: Any) -> AsyncIterator[ModelResponse]:
            for chunk in mock_chunks:
                yield chunk

        mock_acompletion_called_with: dict[str, Any] = {}

        async def mock_acompletion(**kwargs: Any) -> AsyncIterator[ModelResponse]:
            nonlocal mock_acompletion_called_with
            mock_acompletion_called_with = kwargs
            return mock_stream()

        with mock.patch("litellm.acompletion", new=mock_acompletion):
            provider = LiteLLMProvider(TEST_CONFIG)
            chunks = [chunk async for chunk in provider.complete_stream(TEST_MESSAGES)]
            assert mock_acompletion_called_with["stream"] is True
            assert len(chunks) == 2  # noqa: PLR2004
            assert chunks[0].content == "Test "
            assert chunks[1].content == "response"

    @pytest.mark.asyncio
    async def test_error_handling(self) -> None:
        """Test error handling."""

        async def mock_acompletion(**kwargs: Any) -> ModelResponse:
            raise litellm.APIError(
                status_code=1,
                message="API Error",
                llm_provider="test",
                model="test/model",
            )

        with mock.patch("litellm.acompletion", new=mock_acompletion):
            provider = LiteLLMProvider(TEST_CONFIG)
            with pytest.raises(LLMError):
                await provider.complete(TEST_MESSAGES)

    def test_model_capabilities(self) -> None:
        """Test model capabilities detection."""
        provider = LiteLLMProvider(TEST_CONFIG)
        model_info = provider.model_info

        assert model_info.supports_function_calling
        assert "temperature" in model_info.supported_openai_params
        assert "stream" in model_info.supported_openai_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
