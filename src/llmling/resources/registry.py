"""Registry for managing resources."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import logfire

from llmling.config.models import (
    CallableResource,
    CLIResource,
    ImageResource,
    PathResource,
    Resource,
    SourceResource,
    TextResource,
)
from llmling.core import exceptions
from llmling.core.baseregistry import BaseRegistry
from llmling.core.log import get_logger
from llmling.resources.watching import ResourceWatcher


if TYPE_CHECKING:
    from llmling.processors.registry import ProcessorRegistry
    from llmling.resources.loaders.registry import ResourceLoaderRegistry
    from llmling.resources.models import LoadedResource


logger = get_logger(__name__)


class ResourceRegistry(BaseRegistry[str, Resource]):
    """Registry for managing configured resources."""

    def __init__(
        self,
        loader_registry: ResourceLoaderRegistry,
        processor_registry: ProcessorRegistry,
    ) -> None:
        """Initialize registry with required dependencies."""
        super().__init__()
        self.loader_registry = loader_registry
        self.processor_registry = processor_registry
        # Cache by URI instead of name for consistency
        self._cache: dict[str, LoadedResource] = {}
        self._last_loaded: dict[str, datetime] = {}
        self.watcher = ResourceWatcher(self)

    def register(self, key: str, item: Resource | Any, replace: bool = False) -> None:
        """Register an item."""
        # Existing registration logic...
        super().register(key, item, replace)

        # Set up watching if needed
        if self._items[key].is_watched():
            self.watcher.add_watch(key, self._items[key])

    def __delitem__(self, key: str) -> None:
        """Remove item and its watch if present."""
        if key in self._items:
            # Remove watch first
            self.watcher.remove_watch(key)
        super().__delitem__(key)

    async def startup(self) -> None:
        """Initialize registry and start watcher."""
        await super().startup()
        # Start watcher
        await self.watcher.start()
        # Set up watches for existing resources
        for name, resource in self._items.items():
            if resource.is_watched():
                self.watcher.add_watch(name, resource)

    async def shutdown(self) -> None:
        """Cleanup registry and stop watcher."""
        # Stop watcher first
        await self.watcher.stop()
        # Then regular shutdown
        await super().shutdown()

    @property
    def _error_class(self) -> type[exceptions.ResourceError]:
        """Error class to use for this registry."""
        return exceptions.ResourceError

    def _validate_item(self, item: Any) -> Resource:
        """Validate and possibly transform items.

        Args:
            item: Item to validate

        Returns:
            Validated Resource

        Raises:
            ResourceError: If item is invalid
        """
        try:
            match item:
                # Match against concrete resource types
                case (
                    PathResource()
                    | TextResource()
                    | CLIResource()
                    | SourceResource()
                    | CallableResource()
                    | ImageResource()
                ):
                    return item

                case dict() if "resource_type" in item:
                    # Map resource_type to appropriate class
                    resource_classes: dict[str, type[Resource]] = {
                        "path": PathResource,
                        "text": TextResource,
                        "cli": CLIResource,
                        "source": SourceResource,
                        "callable": CallableResource,
                        "image": ImageResource,
                    }

                    resource_type = item["resource_type"]
                    if resource_type not in resource_classes:
                        msg = f"Unknown resource type: {resource_type}"
                        raise exceptions.ResourceError(msg)  # noqa: TRY301

                    # Validate using appropriate class
                    return resource_classes[resource_type].model_validate(item)

                case _:
                    msg = f"Invalid resource type: {type(item)}"
                    raise exceptions.ResourceError(msg)  # noqa: TRY301

        except Exception as exc:
            if isinstance(exc, exceptions.ResourceError):
                raise
            msg = f"Failed to validate resource: {exc}"
            raise exceptions.ResourceError(msg) from exc

    def get_uri(self, name: str) -> str:
        """Get URI for a resource by name."""
        resource = self[name]
        loader = self.loader_registry.get_loader(resource)
        return loader.create_uri(name=name)

    @logfire.instrument("Loading resource {name}")
    async def load(self, name: str, *, force_reload: bool = False) -> LoadedResource:
        """Load a resource by name."""
        try:
            resource = self[name]
            uri = self.get_uri(name)

            # Check cache unless force reload
            if not force_reload and uri in self._cache:
                return self._cache[uri]

            # Get loader and initialize with context
            loader = self.loader_registry.get_loader(resource)
            loader = loader.create(resource, name)  # Create with named context

            loaded = await loader.load(
                context=loader.context,  # Pass the context we created
                processor_registry=self.processor_registry,
            )

            # Ensure the URI is set correctly
            if loaded.metadata.uri != uri:
                logger.warning(
                    "Loader returned different URI than expected: %s != %s",
                    loaded.metadata.uri,
                    uri,
                )
                loaded.metadata.uri = uri

            # Update cache using URI
            self._cache[uri] = loaded
            self._last_loaded[uri] = datetime.now()
        except KeyError as exc:
            msg = f"Resource not found: {name}"
            raise exceptions.ResourceError(msg) from exc
        except Exception as exc:
            msg = f"Failed to load resource {name}: {exc}"
            raise exceptions.ResourceError(msg) from exc
        else:
            return loaded

    async def load_by_uri(self, uri: str) -> LoadedResource:
        """Load a resource by URI."""
        try:
            # Find loader that supports this URI
            loader = self.loader_registry.find_loader_for_uri(uri)

            # Get resource name from URI
            name = loader.get_name_from_uri(uri)

            # Load resource
            return await self.load(name)

        except Exception as exc:
            msg = f"Failed to load resource from URI {uri}"
            raise exceptions.ResourceError(msg) from exc

    def invalidate(self, name: str) -> None:
        """Invalidate cache for a resource."""
        try:
            uri = self.get_uri(name)
            self._cache.pop(uri, None)
            self._last_loaded.pop(uri, None)
        except Exception:
            logger.exception("Failed to invalidate resource: %s", name)

    def clear_cache(self) -> None:
        """Clear all cached resources."""
        self._cache.clear()
        self._last_loaded.clear()
