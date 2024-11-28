"""Path context loader implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from upath import UPath

from llmling.config.models import PathResource
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.resources.base import ResourceLoader, create_loaded_resource


if TYPE_CHECKING:
    from llmling.processors.registry import ProcessorRegistry
    from llmling.resources.models import LoadedResource


logger = get_logger(__name__)


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
        """Handle file:/// URIs properly."""
        try:
            if not cls.supports_uri(uri):
                msg = f"Unsupported URI scheme: {uri}"
                raise exceptions.LoaderError(msg)  # noqa: TRY301

            # Extract path part after file://
            _, path = uri.split("file://", 1)

            # Handle absolute paths (three slashes)
            if path.startswith("///"):
                path = path[3:]

            # Remove ./ and ../ prefixes
            path = path.lstrip("/")
            while path.startswith(("./", "../")):
                if path.startswith("./"):
                    path = path[2:]
                elif path.startswith("../"):
                    path = path[3:]

            # Handle Windows drive letter (e.g., C:/ or D:/)
            if len(path) >= 2 and path[1] == ":":  # noqa: PLR2004
                path = path[2:]  # Skip drive letter and colon

            # Normalize separators and remove any leading slashes
            return path.replace("\\", "/").lstrip("/")

        except Exception as exc:
            msg = f"Invalid file URI: {uri}"
            raise exceptions.LoaderError(msg) from exc

    @classmethod
    def get_uri_template(cls) -> str:
        """File URIs need three slashes for absolute paths."""
        return "file:///{name}"

    @classmethod
    def create_uri(cls, *, name: str) -> str:
        """Handle file paths properly."""
        normalized = name.replace("\\", "/").lstrip("/")
        return cls.get_uri_template().format(name=normalized)

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

            return create_loaded_resource(
                content=content,
                source_type="path",
                uri=self.create_uri(name=name),
                mime_type=self.supported_mime_types[0],
                name=resource.description or path.name,
                description=resource.description,
                additional_metadata={
                    "type": "path",
                    "path": str(path),
                    "scheme": path.protocol,
                },
            )
        except Exception as exc:
            msg = f"Failed to load content from {resource.path}"
            raise exceptions.LoaderError(msg) from exc


if __name__ == "__main__":
    uri = PathResourceLoader.create_uri(name="/path/to/file.txt")
    print(uri)  # file:///path/to/file.txt
