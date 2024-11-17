"""Common type definitions for llmling."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class SupportsStr(Protocol):
    """Protocol for objects that can be converted to string."""

    def __str__(self) -> str: ...


class ProcessingStep(BaseModel):
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
