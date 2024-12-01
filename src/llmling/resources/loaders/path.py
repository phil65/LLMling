from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from upath import UPath

from llmling.config.models import PathResource
from llmling.core import exceptions
from llmling.resources.base import ResourceLoader, create_loaded_resource
from llmling.utils import paths


if TYPE_CHECKING:
    from llmling.processors.registry import ProcessorRegistry
    from llmling.resources.models import LoadedResource


class PathResourceLoader(ResourceLoader[PathResource]):
    """Loads context from files or URLs."""

    context_class = PathResource
    uri_scheme = "file"
    supported_mime_types: ClassVar[list[str]] = [
        "text/plain",
        "application/json",
        "text/markdown",
        "text/yaml",
    ]

    @classmethod
    def get_name_from_uri(cls, uri: str) -> str:
        """Extract the normalized path from a URI."""
        try:
            return paths.uri_to_path(uri)
        except ValueError as exc:
            msg = f"Invalid URI: {uri}"
            raise exceptions.LoaderError(msg) from exc

    @classmethod
    def create_uri(cls, *, name: str) -> str:
        """Create a URI from a path."""
        try:
            return paths.path_to_uri(name)
        except ValueError as exc:
            msg = f"Failed to create URI from {name}"
            raise exceptions.LoaderError(msg) from exc

    async def _load_impl(
        self,
        resource: PathResource,
        name: str,
        processor_registry: ProcessorRegistry | None,
    ) -> LoadedResource:
        """Load content from a file or URL."""
        try:
            path = UPath(resource.path)
            content = path.read_text("utf-8")

            if processor_registry and (procs := resource.processors):
                processed = await processor_registry.process(content, procs)
                content = processed.content
            meta = {"type": "path", "path": str(path), "scheme": path.protocol}
            return create_loaded_resource(
                content=content,
                source_type="path",
                uri=self.create_uri(name=name),
                mime_type=self.supported_mime_types[0],
                name=resource.description or path.name,
                description=resource.description,
                additional_metadata=meta,
            )
        except Exception as exc:
            msg = f"Failed to load content from {resource.path}"
            raise exceptions.LoaderError(msg) from exc


if __name__ == "__main__":
    uri = PathResourceLoader.create_uri(name="/path/to/file.txt")
    print(uri)  # file:///path/to/file.txt
