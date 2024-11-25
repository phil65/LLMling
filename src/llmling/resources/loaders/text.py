from __future__ import annotations

from typing import TYPE_CHECKING

from llmling.config.models import TextResource
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.resources.base import ResourceLoader, create_loaded_resource


if TYPE_CHECKING:
    from llmling.processors.registry import ProcessorRegistry
    from llmling.resources.models import LoadedResource


logger = get_logger(__name__)


class TextResourceLoader(ResourceLoader[TextResource]):
    """Loads context from raw text."""

    context_class = TextResource
    uri_scheme = "text"
    supported_mime_types = ["text/plain"]

    async def load(
        self,
        context: TextResource,
        processor_registry: ProcessorRegistry,
    ) -> LoadedResource:
        context_to_use = context or self.context
        if not context_to_use:
            msg = "No context provided"
            raise exceptions.LoaderError(msg)

        try:
            content = context_to_use.content
            if procs := context_to_use.processors:
                processed = await processor_registry.process(content, procs)
                content = processed.content

            name = context_to_use.description or "text-content"
            return create_loaded_resource(
                content=content,
                source_type="text",
                uri=self.create_uri(name=name),
                mime_type=self.supported_mime_types[0],
                name=context_to_use.description,
                description=context_to_use.description,
            )
        except Exception as exc:
            logger.exception("Failed to load text content")
            msg = "Failed to load text content"
            raise exceptions.LoaderError(msg) from exc
