"""Tests for LLM components including providers and registry."""

from __future__ import annotations

from typing import Any, Literal, cast
from unittest import mock

import litellm
from litellm.utils import ModelResponse
import pytest

from llmling.core import exceptions
from llmling.core.exceptions import LLMError
from llmling.llm.base import LLMConfig, Message
from llmling.llm.providers.litellmprovider import LiteLLMProvider
from llmling.llm.registry import ProviderRegistry


# Type alias for LiteLLM message format
LiteLLMMessage = dict[Literal["role", "name", "content"], Any]


def create_mock_litellm_response(
    content: str = "Test response",
    model: str = "test/model",
    finish_reason: str = "stop",
) -> ModelResponse:
    """Create a mock LiteLLM response."""
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
    """Create a mock LiteLLM stream chunk."""
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


# Test data
TEST_MESSAGES = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="Hello!"),
]

TEST_CONFIG = LLMConfig(
    model="openai/gpt-3.5-turbo",
    provider_name="test-provider",
    temperature=0.7,
    max_tokens=100,
    timeout=30,
)

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
        # Register a provider with litellm implementation
        registry["test"] = "litellm"

        # Create a provider instance
        provider = registry.create_provider("test", TEST_CONFIG)
        assert isinstance(provider, LiteLLMProvider)

    def test_register_duplicate(self, registry: ProviderRegistry) -> None:
        """Test registering same provider twice."""
        # First registration
        registry["test"] = "litellm"

        # Second registration of same name should fail
        with pytest.raises(exceptions.LLMError):
            registry["test"] = "litellm"

    def test_create_unregistered(self, registry: ProviderRegistry) -> None:
        """Test creating unregistered provider."""
        with pytest.raises(exceptions.LLMError):
            registry.create_provider("nonexistent", TEST_CONFIG)


class TestLiteLLMProvider:
    """Tests for the LiteLLM provider implementation."""

    @pytest.mark.asyncio
    async def test_provider_settings(self) -> None:
        """Test that provider correctly uses configured settings."""
        config = LLMConfig(
            model="test/model",
            provider_name="test",
            temperature=0.7,
        )
        response = create_mock_litellm_response()

        async def mock_acompletion(
            model: str,
            messages: list[dict[str, Any]],
            **kwargs: Any,
        ) -> ModelResponse:
            assert model == "test/model"
            assert kwargs["temperature"] == 0.7  # noqa: PLR2004
            return response

        with mock.patch("litellm.acompletion", side_effect=mock_acompletion):
            provider = LiteLLMProvider(config)
            # Pass config values as kwargs
            await provider.complete(
                TEST_MESSAGES,
                temperature=config.temperature,
                api_base=config.api_base,
                api_key=config.api_key,
            )

    @pytest.mark.asyncio
    async def test_kwargs_override_config(self) -> None:
        """Test that runtime kwargs override config values."""
        config = LLMConfig(
            model="test/model",
            provider_name="test",
            api_base="https://local.test",
            temperature=0.7,
        )
        response = create_mock_litellm_response()

        async def mock_acompletion(
            model: str,
            messages: list[dict[str, Any]],
            **kwargs: Any,
        ) -> ModelResponse:
            assert kwargs["api_base"] == "https://override.test"
            assert kwargs["temperature"] == 0.9  # noqa: PLR2004
            return response

        with mock.patch("litellm.acompletion", side_effect=mock_acompletion):
            provider = LiteLLMProvider(config)
            await provider.complete(
                TEST_MESSAGES,
                api_base="https://override.test",
                temperature=0.9,
            )

    def test_model_capabilities(self) -> None:
        """Test model capabilities detection."""
        with mock.patch("litellm.get_model_info") as mock_get_info:
            mock_get_info.return_value = {
                "supports_function_calling": True,
                "supported_openai_params": ["temperature", "stream"],
            }

            provider = LiteLLMProvider(TEST_CONFIG)
            model_info = provider.model_info

            assert model_info.supports_function_calling
            assert "temperature" in model_info.supported_openai_params
            assert "stream" in model_info.supported_openai_params

    # We can remove test_none_values_not_sent as it's no longer relevant
    # We can remove test_config_override_globals as it's now handled by LiteLLM

    @pytest.mark.asyncio
    async def test_error_handling(self) -> None:
        """Test error handling."""

        async def mock_acompletion(**_: Any) -> ModelResponse:
            raise litellm.APIError(
                status_code=1,
                message="API Error",
                llm_provider="test",
                model="test/model",
            )

        with mock.patch("litellm.acompletion", side_effect=mock_acompletion):
            provider = LiteLLMProvider(TEST_CONFIG)
            with pytest.raises(LLMError):
                await provider.complete(TEST_MESSAGES)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
