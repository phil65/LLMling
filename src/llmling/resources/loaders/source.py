"""Loader for Python source code."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, ClassVar

from llmling.config.models import SourceResource
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.resources.base import ResourceLoader, create_loaded_resource
from llmling.utils import importing


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling.processors.registry import ProcessorRegistry
    from llmling.resources.models import LoadedResource


logger = get_logger(__name__)


class SourceResourceLoader(ResourceLoader[SourceResource]):
    """Loads context from Python source code."""

    context_class = SourceResource
    uri_scheme = "python"
    supported_mime_types: ClassVar[list[str]] = ["text/x-python"]

    async def _load_impl(
        self,
        resource: SourceResource,
        name: str,
        processor_registry: ProcessorRegistry | None,
    ) -> AsyncIterator[LoadedResource]:
        """Load Python source content."""
        try:
            content = importing.get_module_source(
                resource.import_path,
                recursive=resource.recursive,
                include_tests=resource.include_tests,
            )

            if processor_registry and (procs := resource.processors):
                processed = await processor_registry.process(content, procs)
                content = processed.content
            meta = {"import_path": resource.import_path, "recursive": resource.recursive}
            yield create_loaded_resource(
                content=content,
                source_type="source",
                uri=self.create_uri(name=name),
                mime_type="text/x-python",
                name=resource.description or resource.import_path,
                description=resource.description,
                additional_metadata=meta,
            )
        except Exception as exc:
            msg = f"Failed to load source from {resource.import_path}"
            raise exceptions.LoaderError(msg) from exc

    async def get_completions(
        self,
        current_value: str,
        argument_name: str | None = None,
        **options: Any,
    ) -> list[str]:
        """Get Python import path completions."""
        try:
            return [name for name in sys.modules if name.startswith(current_value or "")]
        except Exception:
            logger.exception("Import completion failed")
            return []
