"""Path context loader implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import logfire
from upath import UPath

from llmling.config.models import PathResource
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.resources.base import ResourceLoader
from llmling.resources.models import LoadedResource


if TYPE_CHECKING:
    from llmling.processors.registry import ProcessorRegistry


logger = get_logger(__name__)


class PathResourceLoader(ResourceLoader[PathResource]):
    """Loads context from files or URLs."""

    context_class = PathResource

    @logfire.instrument("Loading context from path {context.path}")
    async def load(
        self,
        context: PathResource,
        processor_registry: ProcessorRegistry,
    ) -> LoadedResource:
        """Load content from a file or URL.

        Args:
            context: Path context configuration
            processor_registry: Registry of available processors

        Returns:
            Loaded and processed context

        Raises:
            LoaderError: If loading fails
        """
        try:
            path = UPath(context.path)
            content = path.read_text("utf-8")

            if procs := context.processors:
                processed = await processor_registry.process(content, procs)
                content = processed.content
            meta = {
                "type": "path",
                "path": str(path),
                "size": len(content),
                "scheme": path.protocol,
            }
            return LoadedResource(content=content, source_type="path", metadata=meta)
        except Exception as exc:
            msg = f"Failed to load content from {context.path}"
            logger.exception(msg)
            raise exceptions.LoaderError(msg) from exc
