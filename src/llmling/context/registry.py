"""Registry for context loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
        self._loaders: dict[str, type[ContextLoader[Any]]] = {}

    def register(self, loader_cls: type[ContextLoader[Any]]) -> None:
        """Register a loader using its inferred context_type.

        Args:
            loader_cls: The loader class to register

        Raises:
            LoaderError: If loader is already registered
        """
        # Create instance to get context_type
        loader = loader_cls()
        context_type: str = loader.context_type

        if context_type in self._loaders:
            msg = f"Loader already registered for type: {context_type}"
            raise exceptions.LoaderError(msg)

        logger.debug(
            "Registering loader %s for type %s",
            loader_cls.__name__,
            context_type,
        )
        self._loaders[context_type] = loader_cls

    def get_loader(self, context: Context) -> ContextLoader[Any]:
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
