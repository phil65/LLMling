"""Base classes for context loading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import os
import re
from typing import TYPE_CHECKING, Any, ClassVar, cast, overload

from llmling.completions.protocols import CompletionProvider
from llmling.config.models import BaseResource
from llmling.core import exceptions
from llmling.core.descriptors import classproperty
from llmling.core.log import get_logger
from llmling.core.typedefs import MessageContent
from llmling.resources.models import LoadedResource, ResourceMetadata
from llmling.utils import paths


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling.core.typedefs import MessageContentType
    from llmling.processors.registry import ProcessorRegistry


logger = get_logger(__name__)


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
class LoaderContext[TResource: BaseResource]:
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
        return f"{cls_name}(name={self.name!r}, type={resource.type})"


class ResourceLoader[TResource: BaseResource](ABC, CompletionProvider):
    """Base class for resource loaders."""

    context_class: type[TResource]
    uri_scheme: ClassVar[str]
    supported_mime_types: ClassVar[list[str]] = ["text/plain"]
    # Invalid path characters (Windows + Unix)
    invalid_chars_pattern = re.compile(r'[\x00-\x1F<>:"|?*\\]')

    def __init__(self, context: LoaderContext[TResource] | None = None) -> None:
        """Initialize loader with optional context."""
        self.context = context

    async def get_completions(
        self,
        current_value: str,
        argument_name: str | None = None,
        **options: Any,
    ) -> list[str]:
        """Get completions for this resource type. Override to implement."""
        return []

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
    def get_name_from_uri(cls, uri: str) -> str:
        """Extract resource name from URI, ignoring parameters.

        Args:
            uri: URI in format "{scheme}://{name}[?param1=value1]"

        Returns:
            Resource name

        Raises:
            LoaderError: If URI format is invalid
        """
        import urllib.parse

        try:
            if not cls.supports_uri(uri):
                msg = f"Unsupported URI: {uri}"
                raise exceptions.LoaderError(msg)  # noqa: TRY301

            parsed = urllib.parse.urlparse(uri)
            path = parsed.netloc or parsed.path.lstrip("/")

            # Get parts excluding protocol info
            parts = [
                urllib.parse.unquote(str(part))  # URL decode each part
                for part in path.split("/")
                if not paths.is_ignorable_part(str(part))
            ]

            if not parts:
                msg = "Empty path after normalization"
                raise exceptions.LoaderError(msg)  # noqa: TRY301

            # Validate path components
            if any(cls.invalid_chars_pattern.search(part) for part in parts):
                msg = "Invalid characters in path component"
                raise exceptions.LoaderError(msg)  # noqa: TRY301

            # Join with forward slashes and normalize path
            joined = "/".join(parts)
            # Normalize path (resolve .. and .)
            normalized = os.path.normpath(joined).replace("\\", "/")
        except Exception as exc:
            if isinstance(exc, exceptions.LoaderError):
                raise
            msg = f"Invalid URI format: {uri}"
            raise exceptions.LoaderError(msg) from exc
        else:
            return normalized

    @classmethod
    def get_params_from_uri(cls, uri: str) -> dict[str, str]:
        """Extract parameters from URI.

        Args:
            uri: URI containing query parameters

        Returns:
            Dictionary of parameters
        """
        import urllib.parse

        parsed = urllib.parse.urlparse(uri)
        return dict(urllib.parse.parse_qsl(parsed.query))

    def create_uri(self, *, name: str, params: dict[str, str] | None = None) -> str:
        """Create a valid URI for this resource type.

        Args:
            name: Resource name or identifier
            params: Optional URI parameters

        Returns:
            URI in format "{scheme}://{name}[?param1=value1&...]"
        """
        import urllib.parse

        # Remove any existing scheme
        if "://" in name:
            name = name.split("://", 1)[1]

        base_uri = self.get_uri_template().format(name=name)
        if not params:
            return base_uri

        query = urllib.parse.urlencode(params)
        return f"{base_uri}?{query}"

    def __repr__(self) -> str:
        """Show loader type and context."""
        return f"{self.__class__.__name__}(resource_type={self.resource_type!r})"

    @classproperty  # type: ignore
    def resource_type(self) -> str:
        """Infer context type from context class."""
        fields = self.context_class.model_fields  # type: ignore
        return fields["type"].default  # type: ignore

    @overload
    def load(
        self,
        context: LoaderContext[TResource],
        processor_registry: ProcessorRegistry | None = None,
    ) -> AsyncIterator[LoadedResource]: ...

    @overload
    def load(
        self,
        context: TResource,
        processor_registry: ProcessorRegistry | None = None,
    ) -> AsyncIterator[LoadedResource]: ...

    @overload
    def load(
        self,
        context: None = None,
        processor_registry: ProcessorRegistry | None = None,
    ) -> AsyncIterator[LoadedResource]: ...

    async def load(
        self,
        context: LoaderContext[TResource] | TResource | None = None,
        processor_registry: ProcessorRegistry | None = None,
    ) -> AsyncIterator[LoadedResource]:
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
            case None if self.context:
                resource = self.context.resource
                name = self.context.name
            case None:
                msg = "No context provided"
                raise exceptions.LoaderError(msg)
            case _ if isinstance(context, self.context_class):
                resource = context
                name = "unnamed"  # fallback
            case _:
                msg = f"Invalid context type: {type(context)}"
                raise exceptions.LoaderError(msg)

        # Type assertion to ensure resource is of correct type
        assert isinstance(resource, self.context_class), (
            f"Expected {self.context_class}, got {type(resource)}"
        )
        generator = self._load_impl(resource, name, processor_registry)
        # Then yield from the generator
        async for result in generator:
            yield result

    @abstractmethod
    async def _load_impl(
        self,
        resource: TResource,
        name: str,
        processor_registry: ProcessorRegistry | None,
    ) -> AsyncIterator[LoadedResource]:
        """Implementation of actual loading logic."""
        yield NotImplemented  # type: ignore
