"""Common type definitions for llmling."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, Literal, Protocol, TypedDict, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class ImageMetadata(TypedDict, total=False):
    """Metadata for image content."""

    format: str
    mime_type: str
    width: int
    height: int
    channels: int
    mode: str  # PIL image mode
    has_alpha: bool
    file_size: int
    source: str  # Origin of the image


class ImageContent(TypedDict):
    """Structure for image content in messages."""

    type: Literal["image"]
    data: bytes
    metadata: ImageMetadata


class ContentType(Enum):
    """Types of content that can be processed."""

    TEXT = "text"
    IMAGE = "image"
    MULTI = "multimodal"


T = TypeVar("T", str, bytes, list[Any])


class Content[T]:
    """Generic content container."""

    type: ContentType
    data: T
    metadata: dict[str, Any]


ContentData = str | bytes | list[Any]


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


T = TypeVar("T")
ProcessorCallable = Callable[[str, Any], str | Awaitable[str]]
ContentType = str | SupportsStr
MetadataDict = dict[str, Any]
