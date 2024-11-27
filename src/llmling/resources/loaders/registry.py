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
from llmling.resources.loaders.path import PathResourceLoader
from llmling.resources.loaders.text import TextResourceLoader


if TYPE_CHECKING:
    from llmling.config.models import Resource


logger = get_logger(__name__)


class ResourceLoaderRegistry(BaseRegistry[str, ResourceLoader[Any]]):
    """Registry for context loaders."""

    # def __init__(self) -> None:
    #     """Initialize with default loaders."""
    #     super().__init__()
    #     # Always register defaults
    #     self.register("image", ImageResourceLoader)
    #     self.register("path", PathResourceLoader)
    #     self.register("text", TextResourceLoader)
    #     self.register("cli", CLIResourceLoader)
    #     self.register("source", SourceResourceLoader)
    #     self.register("callable", CallableResourceLoader)

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
        match item:
            case str() if "\n" in item:
                return TextResourceLoader.create(
                    resource=TextResource(content=item),
                    name="inline-text",
                )
            case str() | os.PathLike() if upath.UPath(item).exists():
                return PathResourceLoader.create(
                    resource=PathResource(path=str(item)),
                    name=upath.UPath(item).name,
                )

            case type() if issubclass(item, ResourceLoader):
                return item()

            case ResourceLoader():
                return item
            case _:
                msg = f"Invalid context loader type: {type(item)}"
                raise exceptions.LoaderError(msg)

    def get_loader(self, context: Resource) -> ResourceLoader[Any]:
        """Get a loader instance for a context type."""
        return self.get(context.resource_type)
