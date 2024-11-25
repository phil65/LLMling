from __future__ import annotations

from typing import TYPE_CHECKING

from llmling.config.models import TextContext
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.resources.base import ResourceLoader
from llmling.resources.models import LoadedResource


if TYPE_CHECKING:
    from llmling.processors.registry import ProcessorRegistry


logger = get_logger(__name__)


class TextResourceLoader(ResourceLoader[TextContext]):
    """Loads context from raw text."""

    context_class = TextContext

    async def load(
        self,
        context: TextContext,
        processor_registry: ProcessorRegistry,
    ) -> LoadedResource:
        """Load content from raw text.

        Args:
            context: Text context configuration
            processor_registry: Registry of available processors

        Returns:
            Loaded and processed context

        Raises:
            LoaderError: If loading fails or context type is invalid
        """
        # Use provided context or stored context
        context_to_use = context or self.context
        if not context_to_use:
            msg = "No context provided"
            raise exceptions.LoaderError(msg)

        content = context_to_use.content
        try:
            if procs := context_to_use.processors:
                processed = await processor_registry.process(content, procs)
                content = processed.content
            meta = {"type": "text", "size": len(content)}
            return LoadedResource(content=content, source_type="text", metadata=meta)
        except Exception as exc:
            msg = "Failed to load text content"
            raise exceptions.LoaderError(msg) from exc
