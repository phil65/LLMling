from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from upath import UPath

from llmling.config.models import PathResource
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.resources.base import ResourceLoader, create_loaded_resource
from llmling.utils import paths


logger = get_logger(__name__)

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

    async def get_completions(
        self,
        current_value: str,
        argument_name: str | None = None,
        **options: Any,
    ) -> list[str]:
        """Get path completions."""
        try:
            # Handle both absolute and relative paths
            path = UPath(current_value) if current_value else UPath()
            if not current_value:
                # Show current directory contents for empty input
                matches = list(path.glob("*"))
            else:
                pattern = f"{path}*" if path.is_dir() else f"{path.parent}/*"
                matches = list(UPath().glob(pattern))

            return [str(p) for p in matches if str(p).startswith(current_value or ".")]
        except Exception:
            logger.exception("Path completion failed")
            return []

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
