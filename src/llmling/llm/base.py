"""Base classes and types for LLM integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from llmling.core import exceptions
from llmling.core.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


logger = get_logger(__name__)


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""

    model: str
    provider_name: str  # Key used for provider lookup
    display_name: str = ""  # Human-readable name
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None
    timeout: int = 30
    max_retries: int = 3
    streaming: bool = False


MessageRole = Literal["system", "user", "assistant"]
"""Valid message roles for chat completion."""


class Message(BaseModel):
    """A chat message."""

    role: MessageRole
    content: str

    model_config = ConfigDict(frozen=True)


class CompletionResult(BaseModel):
    """Result from an LLM completion."""

    content: str
    model: str
    finish_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the provider.

        Args:
            config: Provider configuration
        """
        self.config = config

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> CompletionResult:
        """Generate a completion for the messages.

        Args:
            messages: List of messages for chat completion
            **kwargs: Additional provider-specific parameters

        Returns:
            Completion result

        Raises:
            LLMError: If completion fails
        """

    @abstractmethod
    async def complete_stream(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[CompletionResult]:
        """Generate a streaming completion for the messages.

        Args:
            messages: List of messages for chat completion
            **kwargs: Additional provider-specific parameters

        Yields:
            Streamed completion results

        Raises:
            LLMError: If completion fails
        """

    async def validate_response(self, result: CompletionResult) -> None:
        """Validate completion result.

        Args:
            result: Completion result to validate

        Raises:
            LLMError: If validation fails
        """
        if not result.content:
            msg = "Empty response from LLM"
            raise exceptions.LLMError(msg)


class RetryableProvider(LLMProvider):
    """LLM provider with retry support."""

    async def complete(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> CompletionResult:
        """Generate a completion with retry support."""
        retries = 0
        last_error = None

        while retries <= self.config.max_retries:
            try:
                result = await self._complete_impl(messages, **kwargs)
                await self.validate_response(result)
                return result
            except Exception as exc:
                last_error = exc
                retries += 1
                if retries <= self.config.max_retries:
                    await self._handle_retry(exc, retries)
                    continue
                break

        msg = f"Failed after {retries} retries"
        raise exceptions.LLMError(msg) from last_error

    async def complete_stream(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[CompletionResult]:
        """Generate a streaming completion with retry support."""
        retries = 0
        last_error = None

        while retries <= self.config.max_retries:
            try:
                async for result in self._complete_stream_impl(messages, **kwargs):
                    await self.validate_response(result)
                    yield result
                return
            except Exception as exc:
                last_error = exc
                retries += 1
                if retries <= self.config.max_retries:
                    await self._handle_retry(exc, retries)
                    continue
                break

        msg = f"Failed after {retries} retries"
        raise exceptions.LLMError(msg) from last_error

    @abstractmethod
    async def _complete_impl(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> CompletionResult:
        """Implement actual completion logic."""

    @abstractmethod
    async def _complete_stream_impl(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[CompletionResult]:
        """Implement actual streaming completion logic."""

    async def _handle_retry(self, error: Exception, attempt: int) -> None:
        """Handle retry after error.

        Args:
            error: The error that triggered the retry
            attempt: The retry attempt number
        """
        logger.warning(
            "Attempt %d failed, retrying: %s",
            attempt,
            error,
        )
        # Implement exponential backoff
        await asyncio.sleep(2**attempt)
