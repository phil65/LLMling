"""Registry for context loaders."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import upath

from llmling.config.models import PathContext, TextContext
from llmling.context.base import ContextLoader
from llmling.core import exceptions
from llmling.core.baseregistry import BaseRegistry
from llmling.core.log import get_logger


if TYPE_CHECKING:
    from llmling.config.models import Context


logger = get_logger(__name__)


class ContextLoaderRegistry(BaseRegistry[ContextLoader[Any], str]):
    """Registry for context loaders."""

    @property
    def _error_class(self) -> type[exceptions.LoaderError]:
        return exceptions.LoaderError

    def _validate_item(self, item: Any) -> ContextLoader[Any]:
        """Validate and possibly transform item before registration."""
        match item:
            case str() if "\n" in item:  # Multiline string -> TextContext
                from llmling.context.loaders.text import TextContextLoader

                loader = TextContextLoader()
                loader.context = TextContext(content=item)
                return loader
            case str() | os.PathLike() if upath.UPath(item).exists():
                from llmling.context.loaders.path import PathContextLoader

                loader = PathContextLoader()
                loader.context = PathContext(path=item)
                return loader
            case type() as cls if issubclass(cls, ContextLoader):
                return cls()
            case ContextLoader():
                return item
            case _:
                msg = f"Invalid context loader type: {type(item)}"
                raise exceptions.LoaderError(msg)

    def get_loader(self, context: Context) -> ContextLoader[Any]:
        """Get a loader instance for a context type."""
        return self.get(context.context_type)
