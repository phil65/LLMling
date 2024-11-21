"""Registry for context loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
            case type() as cls if issubclass(cls, ContextLoader):
                return cls()
            case _ if isinstance(item, ContextLoader):
                return item
            case _:
                error_msg = f"Invalid context loader type: {type(item)}"
                raise exceptions.LoaderError(error_msg)

    # Backward compatibility methods
    def register(self, loader_cls: type[ContextLoader[Any]]) -> None:  # type: ignore
        """Register a loader using its inferred context_type."""
        super().register(loader_cls.context_type, loader_cls)

    def get_loader(self, context: Context) -> ContextLoader[Any]:
        """Get a loader instance for a context."""
        return self.get(context.context_type)
