"""Registry for context loaders."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import upath

from llmling.config.models import PathContext, TextContext
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

    def _validate_item(self, item: Any) -> ResourceLoader[Any]:
        """Validate and possibly transform item before registration."""
        match item:
            case str() if "\n" in item:  # Multiline string -> TextContext
                from llmling.resources.loaders.text import TextResourceLoader

                return TextResourceLoader(TextContext(content=item))
            case str() | os.PathLike() if upath.UPath(item).exists():
                from llmling.resources.loaders.path import PathResourceLoader

                return PathResourceLoader(PathContext(path=str(item)))
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
