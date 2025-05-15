"""Base classes for content processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, Field
from schemez import Schema

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.utils import calling


if TYPE_CHECKING:
    from llmling.resources.models import ProcessingContext


logger = get_logger(__name__)


class ProcessorConfig(Schema):
    """Configuration for content processors."""

    name: str | None = None
    description: str | None = None
    import_path: str
    async_execution: bool = False
    timeout: float | None = None
    cache_results: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessorResult(Schema):
    """Result of processing content."""

    content: str
    original_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class BaseProcessor(ABC):
    """Base class for all processors."""

    def __init__(self) -> None:
        """Initialize processor."""
        self._initialized = False

    async def startup(self) -> None:
        """Initialize processor resources.

        Override this method if the processor needs async initialization.
        """
        self._initialized = True

    async def shutdown(self) -> None:
        """Clean up processor resources.

        Override this method if the processor needs cleanup.
        """
        self._initialized = False

    @abstractmethod
    async def process(self, context: ProcessingContext) -> ProcessorResult:
        """Process content with given context."""


class Processor(BaseProcessor):
    """Content processor that executes a callable."""

    def __init__(self, config: ProcessorConfig) -> None:
        super().__init__()
        self.config = config
        self._callable: Any = None

    async def startup(self) -> None:
        """Load the callable during startup."""
        if not self.config.import_path:
            msg = "Import path not configured"
            raise exceptions.ProcessorError(msg)

        try:
            self._callable = calling.import_callable(self.config.import_path)
        except ValueError as exc:
            msg = f"Failed to load callable: {exc}"
            raise exceptions.ProcessorError(msg) from exc

        self._initialized = True

    async def process(self, context: ProcessingContext) -> ProcessorResult:
        """Process content using the configured callable."""
        if not self._initialized or not self._callable:
            msg = "Processor not initialized"
            raise exceptions.ProcessorError(msg)

        try:
            # Execute callable
            result = self._callable(context.current_content, **context.kwargs)

            # Handle async functions
            if calling.is_async_callable(self._callable):
                result = await result

            # Convert result to string
            content = str(result)
            is_async = calling.is_async_callable(self._callable)
            meta = {"function": self.config.import_path, "is_async": is_async}
            orig = context.original_content
            return ProcessorResult(content=content, original_content=orig, metadata=meta)
        except Exception as exc:
            msg = f"Processing failed: {exc}"
            raise exceptions.ProcessorError(msg) from exc

    async def shutdown(self) -> None:
        """Clean up resources."""
        self._initialized = False
        self._callable = None
