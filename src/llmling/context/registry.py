"""Registry for context loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling.core import exceptions
from llmling.core.log import get_logger


if TYPE_CHECKING:
    from llmling.config.models import Context
    from llmling.context.base import ContextLoader


logger = get_logger(__name__)


class ContextLoaderRegistry:
    """Registry for context loaders."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._loaders: dict[str, type[ContextLoader]] = {}

    def register(self, type_name: str, loader_cls: type[ContextLoader]) -> None:
        """Register a loader for a context type.

        Args:
            type_name: The context type name (e.g., "path", "text")
            loader_cls: The loader class to register
        """
        logger.debug("Registering loader %s for type %s", loader_cls.__name__, type_name)
        self._loaders[type_name] = loader_cls

    def get_loader(self, context: Context) -> ContextLoader:
        """Get a loader instance for a context.

        Args:
            context: The context configuration

        Returns:
            An instance of the appropriate loader

        Raises:
            LoaderError: If no loader is registered for the context type
        """
        try:
            loader_cls = self._loaders[context.context_type]
            return loader_cls()
        except KeyError as exc:
            msg = f"No loader registered for context type: {context.context_type}"
            raise exceptions.LoaderError(msg) from exc
