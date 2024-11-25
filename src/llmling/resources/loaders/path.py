"""Path context loader implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import logfire
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
    supported_mime_types = [
        "text/plain",
        "application/json",
        "text/markdown",
        "text/yaml",
    ]

    @classmethod
    def get_uri_template(cls) -> str:
        """File URIs need three slashes for absolute paths."""
        return "file:///{name}"

    @classmethod
    def create_uri(cls, *, name: str) -> str:
        """Handle file paths properly."""
        normalized = name.replace("\\", "/").lstrip("/")
        return cls.get_uri_template().format(name=normalized)

    @logfire.instrument("Loading context from path {context.path}")
    async def load(
        self,
        context: PathResource,
        processor_registry: ProcessorRegistry,
    ) -> LoadedResource:
        """Load content from a file or URL.

        Args:
            context: Path context configuration
            processor_registry: Registry of available processors

        Returns:
            Loaded and processed context

        Raises:
            LoaderError: If loading fails
        """
        try:
            path = UPath(context.path)
            content = path.read_text("utf-8")

            if procs := context.processors:
                processed = await processor_registry.process(content, procs)
                content = processed.content

            return create_loaded_resource(
                content=content,
                source_type="path",
                uri=str(path.as_uri()),
                mime_type=self.supported_mime_types[0],
                name=path.name,
                description=context.description,
                additional_metadata={"path": str(path), "scheme": path.protocol},
            )
        except Exception as exc:
            msg = f"Failed to load content from {context.path}"
            raise exceptions.LoaderError(msg) from exc


if __name__ == "__main__":
    uri = PathResourceLoader.create_uri(path="/path/to/file.txt")
    print(uri)  # file:///path/to/file.txt
