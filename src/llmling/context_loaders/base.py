from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel


if TYPE_CHECKING:
    from llmling.config import Context
    from llmling.processors import ProcessorRegistry


class LoaderError(Exception):
    """Base exception for context loading errors."""


class LoadedContext(BaseModel):
    """Result of loading a context."""

    content: str
    metadata: dict[str, Any] = {}


class ContextLoader(abc.ABC):
    """Base class for context loaders."""

    @abc.abstractmethod
    async def load(
        self,
        context: Context,
        processor_registry: ProcessorRegistry,
    ) -> LoadedContext:
        """Load and process context content."""
