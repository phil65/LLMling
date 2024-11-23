"""Base classes and types for LLM integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, Unpack

import py2openai  # noqa: TC002
from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmling.core import exceptions
from llmling.core.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling.llm.clients.protocol import CompletionParams


logger = get_logger(__name__)


class LLMParameters(BaseModel):
    """Base parameters containing LLM params for API calls (LiteLLM based right now)."""

    # Core LLM parameters
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None
    timeout: int = 30
    streaming: bool = False
    tools: list[py2openai.OpenAIFunctionTool] | None = None
    tool_choice: Literal["none", "auto"] | str | None = None  # noqa: PYI051

    # Vision support
    max_image_size: int | None = None

    # API configuration
    api_base: str | None = None
    api_key: str | None = None
    api_version: str | None = None
    bearer_token: str | None = None

    # Retry and timeout settings
    num_retries: int | None = None
    request_timeout: float | None = None
    force_timeout: float | None = None

    # Caching
    cache: bool | None = None
    cache_key: str | None = None

    # Advanced features
    metadata: dict[str, Any] | None = None
    mock_response: str | None = None
    fallbacks: list[str] | None = None
    context_window_fallbacks: list[str] | None = None
    model_list: list[str] | None = None
    drop_params: bool = False
    add_function_to_prompt: bool = False
    proxy_url: str | None = None
    use_queue: bool = False

    model_config = ConfigDict(frozen=True)


class LLMConfig(LLMParameters):
    """Configuration for LLM providers.

    Extends LiteLLM parameters with additional application-specific configuration.
    """

    # Core identification (application specific)
    model: str
    provider_name: str | None = None
    display_name: str = ""

    model_config = ConfigDict(frozen=True)


MessageRole = Literal["system", "user", "assistant", "tool"]
"""Valid message roles for chat completion."""


class ToolCall(BaseModel):
    """A tool call request from the LLM."""

    id: str  # Required by OpenAI
    name: str
    parameters: dict[str, Any]

    model_config = ConfigDict(frozen=True)


ContentType = Literal["text", "image_url", "image_base64"]


class MessageContent(BaseModel):
    """Content item in a message."""

    type: ContentType = "text"  # Default to text for backward compatibility
    content: str
    alt_text: str | None = None  # For image descriptions

    model_config = ConfigDict(frozen=True)


class Message(BaseModel):
    """A chat message."""

    role: MessageRole
    content: str = ""  # Keep for backward compatibility
    content_items: list[MessageContent] = Field(default_factory=list)
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="before")
    @classmethod
    def ensure_content_items(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Ensure content_items is populated from content if empty."""
        if isinstance(data, dict):
            content = data.get("content", "")
            content_items = data.get("content_items", [])
            # Only create content_items from content if we have content and no items
            if content and not content_items:
                data["content_items"] = [
                    MessageContent(type="text", content=content).model_dump()
                ]
            # Always keep content in sync with first text content item
            elif content_items:
                text_items = [
                    item
                    for item in content_items
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                if text_items:
                    data["content"] = text_items[0]["content"]
        return data


class CompletionResult(BaseModel):
    """Result from an LLM completion."""

    content: str
    model: str
    finish_reason: str | None = None
    tool_calls: list[ToolCall] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # cost information
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_cost: float | None = None

    model_config = ConfigDict(frozen=True)


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the provider.

        Args:
            config: Provider configuration
        """
        self.config = config

    def __repr__(self) -> str:
        """Show provider name and model."""
        return f"{self.__class__.__name__}(model={self.config.model!r})"

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        **kwargs: Unpack[CompletionParams],  # Use TypeVar or Any
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
        *,
        chunk_size: int | None = None,
        **kwargs: Unpack[CompletionParams],  # Use TypeVar or Any
    ) -> AsyncIterator[CompletionResult]:
        """Generate a streaming completion for the messages.

        Args:
            messages: List of messages for chat completion
            chunk_size: Streaming chunk size
            **kwargs: Additional provider-specific parameters

        Yields:
            Streamed completion results

        Raises:
            LLMError: If completion fails
        """
        yield NotImplemented  # pragma: no cover

    async def validate_response(self, result: CompletionResult) -> None:
        """Validate completion result.

        Args:
            result: Completion result to validate

        Raises:
            LLMError: If validation fails
        """
        if not result.content and not result.tool_calls:
            msg = "Empty response from LLM"
            raise exceptions.LLMError(msg)
