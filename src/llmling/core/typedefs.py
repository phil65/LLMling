"""Common type definitions for llmling."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field


MessageContentType = Literal["text", "image_url", "image_base64"]


class SupportsStr(Protocol):
    """Protocol for objects that can be converted to string."""

    def __str__(self) -> str: ...


class ProcessingStep(BaseModel):  # type: ignore[no-redef]
    """Configuration for a processing step."""

    name: str
    parallel: bool = False
    required: bool = True
    kwargs: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class MessageContent(BaseModel):
    """Content item in a message."""

    type: MessageContentType = "text"  # Default to text for backward compatibility
    content: str
    alt_text: str | None = None  # For image descriptions

    model_config = ConfigDict(frozen=True)


T = TypeVar("T")
ProcessorCallable = Callable[[str, Any], str | Awaitable[str]]
ContentType = str | SupportsStr
MetadataDict = dict[str, Any]
