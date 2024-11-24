"""Protocol definitions for LLM clients."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NotRequired,
    Protocol,
    TypedDict,
    TypeVar,
    Unpack,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Unpack

    import py2openai

    from llmling.core import capabilities
    from llmling.llm.base import CompletionResult, Message


T = TypeVar("T")


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients."""

    async def complete(
        self,
        messages: list[Message],
        **kwargs: Unpack[LiteLLMCompletionParams],
    ) -> CompletionResult: ...

    async def stream(
        self,
        messages: list[Message],
        chunk_size: int | None = None,
        **kwargs: Unpack[LiteLLMCompletionParams],
    ) -> AsyncIterator[CompletionResult]: ...

    def get_capabilities(self, model: str) -> capabilities.Capabilities: ...


class CompletionParams(TypedDict, total=False):
    """Parameters specific to completion calls."""

    temperature: float
    max_tokens: int | None
    top_p: float | None
    stop: list[str] | str | None
    presence_penalty: float | None
    frequency_penalty: float | None


class APIParams(TypedDict, total=False):
    """API configuration parameters."""

    timeout: int
    api_key: str | None
    api_base: str | None
    api_version: str | None
    num_retries: int | None
    request_timeout: float | None


class LiteLLMCompletionParams(CompletionParams, APIParams):
    """Complete parameter set for LiteLLM completions."""

    tools: NotRequired[list[py2openai.OpenAIFunctionTool]]
    tool_choice: NotRequired[Literal["none", "auto"] | str]  # noqa: PYI051
    max_image_size: NotRequired[int]
    cache: NotRequired[bool]
    metadata: NotRequired[dict[str, Any]]


class CompletionUsage(TypedDict, total=False):
    """Usage information from completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost: float


class LiteLLMImageUrl(TypedDict):
    """Image URL content in LiteLLM format."""

    url: str


class LiteLLMContent(TypedDict):
    """Content item in LiteLLM message format."""

    type: Literal["text", "image_url"]
    text: str | None
    image_url: LiteLLMImageUrl | None


class LiteLLMMessage(TypedDict):
    """Message in LiteLLM format."""

    role: str
    content: str | list[LiteLLMContent]
    name: NotRequired[str | None]
