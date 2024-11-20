"""Context models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llmling.core.typedefs import Content, ContentData, ContentType


class BaseContext(BaseModel):
    """Base class for all context types."""

    content: Content[ContentData]
    metadata: dict[str, Any] = Field(default_factory=dict)

    # model_config = ConfigDict(frozen=True)
    @property
    def content_type(self) -> ContentType:
        return self.content.type


class ProcessingContext(BaseModel):  # type: ignore[no-redef]
    """Context for processor execution."""

    original_content: Content[ContentData]
    current_content: Content[ContentData]
    metadata: dict[str, Any] = Field(default_factory=dict)
    kwargs: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class LoadedContext(BaseContext):
    """Result of loading and processing a context."""

    source_type: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)
