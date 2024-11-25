from __future__ import annotations

from typing import TYPE_CHECKING

import logfire

from llmling.config.models import SourceResource
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.resources.base import ResourceLoader, create_loaded_resource
from llmling.utils import importing


if TYPE_CHECKING:
    from llmling.processors.registry import ProcessorRegistry
    from llmling.resources.models import LoadedResource


logger = get_logger(__name__)


class SourceResourceLoader(ResourceLoader[SourceResource]):
    """Loads context from Python source code."""

    context_class = SourceResource
    uri_scheme = "python"
    supported_mime_types = ["text/x-python"]

    @logfire.instrument("Loading source code from module {context.import_path}")
    async def load(
        self,
        context: SourceResource,
        processor_registry: ProcessorRegistry,
    ) -> LoadedResource:
        """Load content from Python source.

        Args:
            context: Source context configuration
            processor_registry: Registry of available processors

        Returns:
            Loaded and processed context

        Raises:
            LoaderError: If source loading fails or context type is invalid
        """
        try:
            content = importing.get_module_source(
                context.import_path,
                recursive=context.recursive,
                include_tests=context.include_tests,
            )

            if procs := context.processors:
                processed = await processor_registry.process(content, procs)
                content = processed.content

            return create_loaded_resource(
                content=content,
                source_type="source",
                uri=self.create_uri(name=context.import_path),
                mime_type="text/x-python",
                name=context.import_path,
                description=context.description,
                additional_metadata={
                    "import_path": context.import_path,
                    "recursive": context.recursive,
                },
            )
        except Exception as exc:
            msg = f"Failed to load source from {context.import_path}"
            raise exceptions.LoaderError(msg) from exc
