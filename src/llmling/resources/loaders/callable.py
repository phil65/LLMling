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

    async def _load_impl(
        self,
        resource: CallableResource,
        name: str,
        processor_registry: ProcessorRegistry | None,
    ) -> LoadedResource:
        """Execute callable and load result."""
        try:
            kwargs = resource.keyword_args
            content = await calling.execute_callable(resource.import_path, **kwargs)

            if processor_registry and (procs := resource.processors):
                processed = await processor_registry.process(content, procs)
                content = processed.content
            meta = {"import_path": resource.import_path, "args": resource.keyword_args}
            return create_loaded_resource(
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
