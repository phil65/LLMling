"""Base classes for context loading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast, overload

from llmling.config.models import BaseResource
from llmling.core import exceptions
from llmling.core.descriptors import classproperty
from llmling.core.log import get_logger
from llmling.core.typedefs import MessageContent, MessageContentType
from llmling.resources.models import LoadedResource, ResourceMetadata


if TYPE_CHECKING:
    from llmling.processors.registry import ProcessorRegistry


logger = get_logger(__name__)

TResource = TypeVar("TResource", bound=BaseResource)


def create_loaded_resource(
    *,
    content: str,
    source_type: str,
    uri: str,
    mime_type: str | None = None,
    name: str | None = None,
    description: str | None = None,
    additional_metadata: dict[str, Any] | None = None,
    content_type: MessageContentType = "text",
    content_items: list[MessageContent] | None = None,
) -> LoadedResource:
    """Create a LoadedResource with all required fields.

    Args:
        content: The main content (for backwards compatibility)
        source_type: Type of source ("text", "path", etc.)
        uri: Resource URI
        mime_type: Content MIME type
        name: Resource name
        description: Resource description
        additional_metadata: Additional metadata
        content_type: Type of content for default content item
        content_items: Optional list of content items (overrides default)
    """
    metadata = ResourceMetadata(
        uri=uri,
        mime_type=mime_type or "text/plain",
        name=name or f"{source_type.title()} resource",
        description=description,
        size=len(content),
        modified=datetime.now().isoformat(),
        extra=additional_metadata or {},
    )

    # Use provided content items or create default text item
    items = content_items or [MessageContent(type=content_type, content=content)]

    return LoadedResource(
        content=content,
        source_type=source_type,
        metadata=metadata,
        content_items=items,
        etag=f"{source_type}-{metadata.size}-{metadata.modified}",
    )


@dataclass
class LoaderContext[TResource]:
    """Context for resource loading.

    Provides all information needed to load and identify a resource.
    """

    resource: TResource
    name: str

    def __repr__(self) -> str:
        """Show context details."""
        # Cast to Protocol to make type checker happy
        resource = cast(BaseResource, self.resource)
        cls_name = self.__class__.__name__
        return f"{cls_name}(name={self.name!r}, type={resource.resource_type})"


class ResourceLoader[TResource](ABC):
    """Base class for resource loaders."""

    context_class: type[TResource]
    uri_scheme: ClassVar[str]
    supported_mime_types: ClassVar[list[str]] = ["text/plain"]

    def __init__(self, context: LoaderContext[TResource] | None = None) -> None:
        """Initialize loader with optional context."""
        self.context = context

    @classmethod
    def create(cls, resource: TResource, name: str) -> ResourceLoader[TResource]:
        """Create a loader instance with named context."""
        return cls(LoaderContext(resource=resource, name=name))

    @classmethod
    def supports_uri(cls, uri: str) -> bool:
        """Check if this loader supports a given URI."""
        return uri.startswith(f"{cls.uri_scheme}://")

    @classmethod  # could be classproperty
    def get_uri_template(cls) -> str:
        """Get the URI template for this resource type."""
        return f"{cls.uri_scheme}://{{name}}"

    @classmethod
    def create_uri(cls, *, name: str) -> str:
        """Create a valid URI for this resource type."""
        return cls.get_uri_template().format(name=name)

    def __repr__(self) -> str:
        """Show loader type and context."""
        return f"{self.__class__.__name__}(resource_type={self.resource_type!r})"

    @classproperty  # type: ignore
    def resource_type(self) -> str:
        """Infer context type from context class."""
        fields = self.context_class.model_fields  # type: ignore
        return fields["resource_type"].default  # type: ignore

    @overload
    async def load(
        self,
        context: LoaderContext[TResource],
        processor_registry: ProcessorRegistry | None = None,
    ) -> LoadedResource: ...

    @overload
    async def load(
        self,
        context: TResource,
        processor_registry: ProcessorRegistry | None = None,
    ) -> LoadedResource: ...

    @overload
    async def load(
        self,
        context: None = None,
        processor_registry: ProcessorRegistry | None = None,
    ) -> LoadedResource: ...

    async def load(
        self,
        context: LoaderContext[TResource] | TResource | None = None,
        processor_registry: ProcessorRegistry | None = None,
    ) -> LoadedResource:
        """Load and process content.

        Args:
            context: Either a LoaderContext, direct Resource, or None (uses self.context)
            processor_registry: Optional processor registry for content processing

        Returns:
            Loaded resource content

        Raises:
            LoaderError: If loading fails
        """
        # Resolve the actual resource and name
        match context:
            case LoaderContext():
                resource = context.resource
                name = context.name
            case self.context_class():
                resource = context
                name = "unnamed"  # fallback
            case None if self.context:
                resource = self.context.resource
                name = self.context.name
            case None:
                msg = "No context provided"
                raise exceptions.LoaderError(msg)
            case _:
                msg = f"Invalid context type: {type(context)}"
                raise exceptions.LoaderError(msg)

        return await self._load_impl(resource, name, processor_registry)

    @abstractmethod
    async def _load_impl(
        self,
        resource: TResource,
        name: str,
        processor_registry: ProcessorRegistry | None,
    ) -> LoadedResource:
        """Implementation of actual loading logic."""
