from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

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

    @classmethod
    def get_name_from_uri(cls, uri: str) -> str:
        """Extract the normalized path from a URI."""
        try:
            if not cls.supports_uri(uri):
                msg = f"Unsupported URI: {uri}"
                raise exceptions.LoaderError(msg)  # noqa: TRY301

            path = UPath(uri)

            # Get parts excluding protocol info
            parts = [part for part in path.parts if not cls._is_ignorable_part(str(part))]

            if not parts:
                msg = "Empty path after normalization"
                raise exceptions.LoaderError(msg)  # noqa: TRY301

            # Validate path components
            for part in parts:
                if cls.invalid_chars_pattern.search(str(part)):
                    msg = f"Invalid characters in path component: {part}"
                    raise exceptions.LoaderError(msg)  # noqa: TRY301

            # Join with forward slashes and normalize consecutive slashes
            joined = "/".join(str(p) for p in parts)
            return re.sub(
                r"/+", "/", joined
            )  # Replace multiple slashes with single slash

        except Exception as exc:
            if isinstance(exc, exceptions.LoaderError):
                raise
            msg = f"Invalid URI: {uri}"
            raise exceptions.LoaderError(msg) from exc

    @classmethod
    def create_uri(cls, *, name: str) -> str:
        """Create a URI from a path or URL."""
        try:
            # Validate path
            if cls.invalid_chars_pattern.search(name):
                msg = f"Invalid characters in path: {name}"
                raise exceptions.LoaderError(msg)  # noqa: TRY301

            # Use UPath for handling
            path = UPath(name)

            # If it already has a protocol, use it as-is
            if path.protocol:
                return str(path)

            # Otherwise, treat as local file
            return f"file:///{str(path).lstrip('/')}"

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
