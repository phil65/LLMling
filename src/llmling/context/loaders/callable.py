from __future__ import annotations

from typing import TYPE_CHECKING

from llmling.config.models import CallableContext
from llmling.context.base import ContextLoader
from llmling.context.models import LoadedContext
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.utils import calling


if TYPE_CHECKING:
    from llmling.config.models import Context
    from llmling.processors.registry import ProcessorRegistry


logger = get_logger(__name__)


class CallableContextLoader(ContextLoader):
    """Loads context from Python callable execution."""

    async def load(
        self,
        context: Context,
        processor_registry: ProcessorRegistry,
    ) -> LoadedContext:
        """Load content from callable execution.

        Args:
            context: Callable context configuration
            processor_registry: Registry of available processors

        Returns:
            Loaded and processed context

        Raises:
            LoaderError: If callable execution fails or context type is invalid
        """
        if not isinstance(context, CallableContext):
            msg = f"Expected CallableContext, got {type(context).__name__}"
            raise exceptions.LoaderError(msg)

        try:
            content = await calling.execute_callable(
                context.import_path,
                **context.keyword_args,
            )

            if context.processors:
                processed = await processor_registry.process(
                    content,
                    context.processors,
                )
                content = processed.content

            return LoadedContext(
                content=content,
                source_type="callable",
                metadata={
                    "type": "callable",
                    "import_path": context.import_path,
                    "size": len(content),
                },
            )
        except Exception as exc:
            msg = f"Failed to execute callable {context.import_path}"
            raise exceptions.LoaderError(msg) from exc
