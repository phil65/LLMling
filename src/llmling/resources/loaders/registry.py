"""Registry for context loaders."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from llmling.config.models import PathResource, TextResource
from llmling.core import exceptions
from llmling.core.baseregistry import BaseRegistry
from llmling.core.log import get_logger
from llmling.resources.base import ResourceLoader
from llmling.resources.loaders.path import PathResourceLoader
from llmling.resources.loaders.text import TextResourceLoader


if TYPE_CHECKING:
    from llmling.config.models import BaseResource


logger = get_logger(__name__)


class ResourceLoaderRegistry(BaseRegistry[str, ResourceLoader[Any]]):
    """Registry for context loaders."""

    def __init__(self, register_default_loaders: bool = False, **kwargs) -> None:
        """Initialize registry.

        Args:
            register_default_loaders: Register default loaders on initialization.
            kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        if register_default_loaders:
            self.register_default_loaders()

    @property
    def _error_class(self) -> type[exceptions.LoaderError]:
        return exceptions.LoaderError

    def get_supported_schemes(self) -> list[str]:
        """Get all supported URI schemes."""
        return [loader.uri_scheme for loader in self._items.values()]

    def register_default_loaders(self) -> None:
        from llmling.resources import (
            CallableResourceLoader,
            CLIResourceLoader,
            PathResourceLoader,
            RepositoryResourceLoader,
            SourceResourceLoader,
            TextResourceLoader,
        )

        self["path"] = PathResourceLoader
        self["text"] = TextResourceLoader
        self["cli"] = CLIResourceLoader
        self["source"] = SourceResourceLoader
        self["callable"] = CallableResourceLoader
        self["repository"] = RepositoryResourceLoader

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
        # Parse scheme from URI string
        try:
            scheme = uri.split("://")[0]
        except IndexError:
            msg = f"Invalid URI format: {uri}"
            raise exceptions.LoaderError(msg) from None

        for loader in self._items.values():
            if loader.uri_scheme == scheme:
                return loader

        msg = f"No loader found for URI scheme: {scheme}"
        raise exceptions.LoaderError(msg)

    def _validate_item(self, item: Any) -> ResourceLoader[Any]:
        """Validate and possibly transform item before registration."""
        from upathtools import to_upath

        match item:
            case str() if "\n" in item:
                resource = TextResource(content=item)
                return TextResourceLoader.create(resource=resource, name="inline-text")
            case str() | os.PathLike() if to_upath(item).exists():
                path_resource = PathResource(path=str(item))
                name = to_upath(item).name
                return PathResourceLoader.create(resource=path_resource, name=name)
            case type() if issubclass(item, ResourceLoader):
                return item()
            case ResourceLoader():
                return item
            case _:
                msg = f"Invalid context loader type: {type(item)}"
                raise exceptions.LoaderError(msg)

    def get_loader(self, context: BaseResource) -> ResourceLoader[Any]:
        """Get a loader instance for a context type."""
        return self.get(context.type)
