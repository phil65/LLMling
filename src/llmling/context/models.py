"""Context models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BaseContext(BaseModel):
    """Base class for all context types."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class ProcessingContext(BaseModel):  # type: ignore[no-redef]
    """Context for processor execution."""

    original_content: str
    current_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    kwargs: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class LoadedContext(BaseContext):
    """Result of loading and processing a context."""

    source_type: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)
