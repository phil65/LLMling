from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from llmling.config.models import CallableResource
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.resources.base import ResourceLoader, create_loaded_resource
from llmling.utils import calling


if TYPE_CHECKING:
    from llmling.processors.registry import ProcessorRegistry
    from llmling.resources.models import LoadedResource


logger = get_logger(__name__)


class CallableResourceLoader(ResourceLoader[CallableResource]):
    """Loads context from Python callable execution."""

    context_class = CallableResource
    uri_scheme = "callable"
    supported_mime_types: ClassVar[list[str]] = ["text/plain"]

    async def load(
        self,
        context: CallableResource,
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

            return create_loaded_resource(
                content=content,
                source_type="callable",
                uri=self.create_uri(name=context.import_path),
                name=context.import_path,
                description=context.description,
                additional_metadata={
                    "import_path": context.import_path,
                    "args": context.keyword_args,
                },
            )
        except Exception as exc:
            msg = f"Failed to execute callable {context.import_path}"
            raise exceptions.LoaderError(msg) from exc
