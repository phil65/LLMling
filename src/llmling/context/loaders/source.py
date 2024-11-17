from __future__ import annotations

from typing import TYPE_CHECKING

import logfire

from llmling.config.models import SourceContext
from llmling.context.base import ContextLoader
from llmling.context.models import LoadedContext
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.utils import importing


if TYPE_CHECKING:
    from llmling.config.models import Context
    from llmling.processors.registry import ProcessorRegistry


logger = get_logger(__name__)


class SourceContextLoader(ContextLoader):
    """Loads context from Python source code."""

    @logfire.instrument("Loading source code from module {context.import_path}")
    async def load(
        self,
        context: Context,
        processor_registry: ProcessorRegistry,
    ) -> LoadedContext:
        """Load content from Python source.

        Args:
            context: Source context configuration
            processor_registry: Registry of available processors

        Returns:
            Loaded and processed context

        Raises:
            LoaderError: If source loading fails or context type is invalid
        """
        if not isinstance(context, SourceContext):
            msg = f"Expected SourceContext, got {type(context).__name__}"
            raise exceptions.LoaderError(msg)

        try:
            content = importing.get_module_source(
                context.import_path,
                recursive=context.recursive,
                include_tests=context.include_tests,
            )

            if context.processors:
                processed = await processor_registry.process(
                    content,
                    context.processors,
                )
                content = processed.content

            return LoadedContext(
                content=content,
                source_type="source",
                metadata={
                    "type": "source",
                    "import_path": context.import_path,
                    "size": len(content),
                },
            )
        except Exception as exc:
            msg = f"Failed to load source from {context.import_path}"
            raise exceptions.LoaderError(msg) from exc
