"""Registry for context loaders."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import upath

from llmling.config.models import PathResource, TextResource
from llmling.core import exceptions
from llmling.core.baseregistry import BaseRegistry
from llmling.core.log import get_logger
from llmling.resources.base import ResourceLoader


if TYPE_CHECKING:
    from llmling.config.models import Context


logger = get_logger(__name__)


class ResourceLoaderRegistry(BaseRegistry[str, ResourceLoader[Any]]):
    """Registry for context loaders."""

    @property
    def _error_class(self) -> type[exceptions.LoaderError]:
        return exceptions.LoaderError

    def get_supported_schemes(self) -> list[str]:
        """Get all supported URI schemes."""
        return [loader.uri_scheme for loader in self._items.values()]

    def get_uri_templates(self) -> list[dict[str, Any]]:
        """Get URI templates for all registered loaders."""
        return [
            {
                "scheme": loader.uri_scheme,
                "template": loader.get_uri_template(),
                "mimeTypes": loader.supported_mime_types,
            }
            for loader in self._items.values()
        ]

    def find_loader_for_uri(self, uri: str) -> ResourceLoader[Any]:
        """Find appropriate loader for a URI."""
        for loader in self._items.values():
            if loader.supports_uri(uri):
                return loader
        msg = f"No loader found for URI scheme: {uri.scheme}"
        raise exceptions.LoaderError(msg)

    def _validate_item(self, item: Any) -> ResourceLoader[Any]:
        """Validate and possibly transform item before registration."""
        match item:
            case str() if "\n" in item:  # Multiline string -> TextResource
                from llmling.resources.loaders.text import TextResourceLoader

                return TextResourceLoader(TextResource(content=item))
            case str() | os.PathLike() if upath.UPath(item).exists():
                from llmling.resources.loaders.path import PathResourceLoader

                return PathResourceLoader(PathResource(path=str(item)))
            case type() as cls if issubclass(cls, ResourceLoader):
                return cls()
            case ResourceLoader():
                return item
            case _:
                msg = f"Invalid context loader type: {type(item)}"
                raise exceptions.LoaderError(msg)

    def get_loader(self, context: Context) -> ResourceLoader[Any]:
        """Get a loader instance for a context type."""
        return self.get(context.context_type)
