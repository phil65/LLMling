from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
import urllib.parse

from upath import UPath

from llmling.config.models import PathResource
from llmling.core import exceptions
from llmling.resources.base import ResourceLoader, create_loaded_resource


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
    def supports_uri(cls, uri: str) -> bool:
        """Check if this loader supports a given URI using upath's protocol system."""
        try:
            # Let UPath handle protocol support
            UPath(uri)
        except (ValueError, NotImplementedError):
            return False
        else:
            return True

    @staticmethod
    def _normalize_path_components(parts: list[str]) -> list[str]:
        """Normalize path components, resolving . and .. entries."""
        result: list[str] = []
        for part in parts:
            match part:
                case "." | "":
                    continue
                case "..":
                    if result:
                        result.pop()
                case _:
                    if not PathResourceLoader._is_ignorable_part(part):
                        result.append(part)
        return result

    @classmethod
    def get_name_from_uri(cls, uri: str) -> str:
        """Extract the normalized path from a URI."""
        try:
            if not uri.startswith("file:///"):
                msg = f"Invalid file URI format: {uri}"
                raise exceptions.LoaderError(msg)  # noqa: TRY301

            # Remove the file:/// prefix
            path = uri[8:]

            # Split into components and decode, including empty parts
            parts = [urllib.parse.unquote(part) for part in path.split("/")]

            # Remove empty parts and normalize
            normalized_parts = cls._normalize_path_components(parts)
            if not normalized_parts:
                msg = "Empty path after normalization"
                raise exceptions.LoaderError(msg)  # noqa: TRY301

            # Validate normalized components
            for part in normalized_parts:
                if cls.invalid_chars_pattern.search(part):
                    msg = f"Invalid characters in path component: {part}"
                    raise exceptions.LoaderError(msg)  # noqa: TRY301

            return "/".join(normalized_parts)

        except Exception as exc:
            if isinstance(exc, exceptions.LoaderError):
                raise
            msg = f"Invalid URI: {uri}"
            raise exceptions.LoaderError(msg) from exc

    @classmethod
    def create_uri(cls, *, name: str) -> str:
        """Create a URI from a path."""
        try:
            # Validate path
            if cls.invalid_chars_pattern.search(name):
                msg = f"Invalid characters in path: {name}"
                raise exceptions.LoaderError(msg)  # noqa: TRY301

            # Normalize path separators and split
            parts = name.replace("\\", "/").split("/")

            # Normalize components
            normalized_parts = cls._normalize_path_components(parts)
            if not normalized_parts:
                msg = "Empty path"
                raise exceptions.LoaderError(msg)  # noqa: TRY301

            # Encode components and create URI
            encoded_parts = [urllib.parse.quote(part) for part in normalized_parts]
            return f"file:///{'/'.join(encoded_parts)}"

        except Exception as exc:
            if isinstance(exc, exceptions.LoaderError):
                raise
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
