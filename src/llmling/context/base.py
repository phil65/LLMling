"""Base classes for context loading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from llmling.core import exceptions
from llmling.core.descriptors import classproperty
from llmling.core.log import get_logger


if TYPE_CHECKING:
    from llmling.config.models import (
        Context,
    )
    from llmling.context.models import LoadedContext
    from llmling.processors.registry import ProcessorRegistry


logger = get_logger(__name__)

TContext = TypeVar("TContext", bound="Context")


class ContextLoader[TContext](ABC):
    """Base class for context loaders with associated context type."""

    context_class: type[TContext]

    @classproperty  # type: ignore
    def context_type(self) -> str:
        """Infer context type from context class."""
        fields = self.context_class.model_fields  # type: ignore
        return fields["context_type"].default  #  type: ignore

    @abstractmethod
    async def load(
        self,
        context: TContext,
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
