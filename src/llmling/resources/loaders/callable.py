"""Callable resource loader implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from llmling.config.models import CallableResource
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.resources.base import ResourceLoader, create_loaded_resource
from llmling.utils import calling


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling.processors.registry import ProcessorRegistry
    from llmling.resources.models import LoadedResource


logger = get_logger(__name__)


class CallableResourceLoader(ResourceLoader[CallableResource]):
    """Loads context from Python callable execution with URI parameter support."""

    context_class = CallableResource
    uri_scheme = "callable"
    supported_mime_types: ClassVar[list[str]] = ["text/plain"]

    async def _load_impl(
        self,
        resource: CallableResource,
        name: str,
        processor_registry: ProcessorRegistry | None,
    ) -> AsyncIterator[LoadedResource]:
        """Execute callable with parameters from URI if present."""
        try:
            # Get base kwargs from resource config
            kwargs = resource.keyword_args.copy()

            # If we have a URI context, extract and merge parameters
            if self.context:
                uri = self.create_uri(name=self.context.name)
                uri_params = self.get_params_from_uri(uri)
                kwargs.update(uri_params)

            # Execute the callable with combined parameters
            content = await calling.execute_callable(resource.import_path, **kwargs)

            if processor_registry and (procs := resource.processors):
                processed = await processor_registry.process(content, procs)
                content = processed.content

            meta = {"import_path": resource.import_path, "args": kwargs}

            yield create_loaded_resource(
                content=content,
                source_type="callable",
                uri=self.create_uri(name=name),
                name=resource.description or resource.import_path,
                description=resource.description,
                additional_metadata=meta,
            )
        except Exception as exc:
            msg = f"Failed to execute callable {resource.import_path}"
            raise exceptions.LoaderError(msg) from exc
