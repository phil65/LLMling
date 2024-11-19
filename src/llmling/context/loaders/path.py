from __future__ import annotations

from typing import TYPE_CHECKING

import logfire
from upath import UPath

from llmling.config.models import PathContext
from llmling.context.base import ContextLoader
from llmling.context.models import LoadedContext
from llmling.core import exceptions
from llmling.core.log import get_logger


if TYPE_CHECKING:
    from llmling.config.models import Context
    from llmling.processors.registry import ProcessorRegistry


logger = get_logger(__name__)


class PathContextLoader(ContextLoader):
    """Loads context from files or URLs."""

    @logfire.instrument("Loading context from path {context.path}")
    async def load(
        self,
        context: Context,
        processor_registry: ProcessorRegistry,
    ) -> LoadedContext:
        """Load content from a file or URL.

        Args:
            context: Path context configuration
            processor_registry: Registry of available processors

        Returns:
            Loaded and processed context

        Raises:
            LoaderError: If loading fails or context type is invalid
        """
        if not isinstance(context, PathContext):
            msg = f"Expected PathContext, got {type(context).__name__}"
            raise exceptions.LoaderError(msg)

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
            return LoadedContext(content=content, source_type="path", metadata=meta)
        except Exception as exc:
            msg = f"Failed to load content from {context.path}"
            logger.exception(msg)
            raise exceptions.LoaderError(msg) from exc
