from __future__ import annotations

from typing import TYPE_CHECKING

from llmling.config.models import CallableContext
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.resources.base import ResourceLoader
from llmling.resources.models import LoadedResource
from llmling.utils import calling


if TYPE_CHECKING:
    from llmling.processors.registry import ProcessorRegistry


logger = get_logger(__name__)


class CallableResourceLoader(ResourceLoader[CallableContext]):
    """Loads context from Python callable execution."""

    context_class = CallableContext

    async def load(
        self,
        context: CallableContext,
        processor_registry: ProcessorRegistry,
    ) -> LoadedResource:
        """Load content from callable execution.

        Args:
            context: Callable context configuration
            processor_registry: Registry of available processors

        Returns:
            Loaded and processed context

        Raises:
            LoaderError: If callable execution fails or context type is invalid
        """
        try:
            content = await calling.execute_callable(
                context.import_path, **context.keyword_args
            )

            if procs := context.processors:
                processed = await processor_registry.process(content, procs)
                content = processed.content
            meta = {
                "type": "callable",
                "import_path": context.import_path,
                "size": len(content),
            }
            return LoadedResource(content=content, source_type="callable", metadata=meta)
        except Exception as exc:
            msg = f"Failed to execute callable {context.import_path}"
            raise exceptions.LoaderError(msg) from exc
