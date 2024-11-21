"""Base classes for context loading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from llmling.core import exceptions
from llmling.core.log import get_logger


if TYPE_CHECKING:
    from llmling.config.models import Context, ContextType
    from llmling.context.models import LoadedContext
    from llmling.processors.registry import ProcessorRegistry


logger = get_logger(__name__)


class ContextLoader(ABC):
    """Base class for context loaders."""

    context_type: ClassVar[ContextType]

    @abstractmethod
    async def load(
        self,
        context: Context,
        processor_registry: ProcessorRegistry,
    ) -> LoadedContext:
        """Load and process context content.

        Args:
            context: The loading-context
            processor_registry: Registry of available processors

        Returns:
            Loaded and processed context

        Raises:
            LoaderError: If loading fails
        """

    async def _process_content(
        self,
        content: str,
        config: Any,
        processor_registry: ProcessorRegistry,
    ) -> str:
        """Process content through configured processors."""
        try:
            # Will be implemented when processors are refactored
            return content
        except Exception as exc:
            msg = "Failed to process content"
            raise exceptions.ProcessorError(msg) from exc
