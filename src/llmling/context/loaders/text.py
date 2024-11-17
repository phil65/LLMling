from __future__ import annotations

from typing import TYPE_CHECKING

from llmling.config.models import TextContext
from llmling.context.base import ContextLoader
from llmling.context.models import LoadedContext
from llmling.core import exceptions
from llmling.core.log import get_logger


if TYPE_CHECKING:
    from llmling.config.models import Context
    from llmling.processors.registry import ProcessorRegistry


logger = get_logger(__name__)


class TextContextLoader(ContextLoader):
    """Loads context from raw text."""

    async def load(
        self,
        context: Context,
        processor_registry: ProcessorRegistry,
    ) -> LoadedContext:
        """Load content from raw text.

        Args:
            context: Text context configuration
            processor_registry: Registry of available processors

        Returns:
            Loaded and processed context

        Raises:
            LoaderError: If loading fails or context type is invalid
        """
        if not isinstance(context, TextContext):
            msg = f"Expected TextContext, got {type(context).__name__}"
            raise exceptions.LoaderError(msg)

        try:
            content = context.content

            if context.processors:
                processed = await processor_registry.process(
                    content,
                    context.processors,
                )
                content = processed.content

            return LoadedContext(
                content=content,
                source_type="text",
                metadata={
                    "type": "text",
                    "size": len(content),
                },
            )
        except Exception as exc:
            msg = "Failed to load text content"
            raise exceptions.LoaderError(msg) from exc
