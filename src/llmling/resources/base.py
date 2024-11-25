"""Base classes for context loading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from llmling.core import exceptions
from llmling.core.descriptors import classproperty
from llmling.core.log import get_logger
from llmling.core.typedefs import MessageContent
from llmling.resources.models import LoadedResource, ResourceMetadata


if TYPE_CHECKING:
    from llmling.config.models import Context
    from llmling.processors.registry import ProcessorRegistry


logger = get_logger(__name__)

TContext = TypeVar("TContext", bound="Context")


def create_loaded_resource(
    *,
    content: str,
    source_type: str,
    uri: str,
    mime_type: str | None = None,
    name: str | None = None,
    description: str | None = None,
    additional_metadata: dict[str, Any] | None = None,
    content_type: str = "text",
) -> LoadedResource:
    """Create a LoadedResource with all required fields."""
    # Ensure URI is valid
    if not uri:
        uri = f"{source_type}://{name or 'unnamed'}"

    # Create base metadata
    metadata = ResourceMetadata(
        uri=uri,
        mime_type=mime_type or "text/plain",
        name=name or f"{source_type.title()} resource",
        description=description,
        size=len(content),
        modified=datetime.now().isoformat(),
        # If we have additional metadata, include it in the initial creation
        **(additional_metadata or {}),
    )

    return LoadedResource(
        content=content,
        source_type=source_type,
        metadata=metadata,
        content_items=[MessageContent(type=content_type, content=content)],
        etag=f"{source_type}-{metadata.size}-{metadata.modified}",
    )


class ResourceLoader[TContext](ABC):
    """Base class for context loaders with associated context type."""

    context_class: type[TContext]
    uri_scheme: ClassVar[str]  # e.g., "file", "text", "git"

    # Optional media types this loader supports
    supported_mime_types: ClassVar[list[str]] = ["text/plain"]

    def __init__(self, context: TContext | None = None) -> None:
        """Initialize loader with optional context.

        Args:
            context: Optional pre-configured context
        """
        self.context = context

    @classmethod
    def supports_uri(cls, uri: AnyUrl) -> bool:
        """Check if this loader supports a given URI."""
        return uri.scheme == cls.uri_scheme

    @classmethod  # could be classproperty
    def get_uri_template(cls) -> str:
        """Get the URI template for this resource type."""
        return f"{cls.uri_scheme}://{{name}}"

    @classmethod
    def create_uri(cls, name: str) -> str:
        """Create a valid URI for this resource type."""
        return cls.get_uri_template().format(name=name)

    def __repr__(self) -> str:
        """Show loader type and context."""
        return f"{self.__class__.__name__}(context_type={self.context_type!r})"

    @classproperty  # type: ignore
    def context_type(self) -> str:
        """Infer context type from context class."""
        fields = self.context_class.model_fields  # type: ignore
        return fields["context_type"].default  # type: ignore

    @abstractmethod
    async def load(
        self,
        context: TContext,
        processor_registry: ProcessorRegistry,
    ) -> LoadedResource:
        """Load and process context content.

        Args:
            context: The loading-context
            processor_registry: Registry of available processors

        Returns:
            Loaded and processed context

        Raises:
            LoaderError: If loading fails
        """

    async def _process_content(
        self,
        content: str,
        config: Any,
        processor_registry: ProcessorRegistry,
    ) -> str:
        """Process content through configured processors."""
        try:
            # Will be implemented when processors are refactored
            return content
        except Exception as exc:
            msg = "Failed to process content"
            raise exceptions.ProcessorError(msg) from exc
