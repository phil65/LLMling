# llmling/core/registry.py
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, MutableMapping
from typing import TYPE_CHECKING, Any, TypeVar


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling.core import exceptions


TItem = TypeVar("TItem")
TKey = TypeVar("TKey", str, int)


class BaseRegistry[TKey, TItem](MutableMapping[TKey, TItem], ABC):
    """Base class for component registries."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._items: dict[TKey, TItem] = {}
        self._initialized = False
        self._configs: dict[TKey, Any] = {}

    @property
    def is_empty(self) -> bool:
        """Check if registry has any items."""
        return not bool(self._items)

    def has_item(self, key: TKey) -> bool:
        """Check if an item is registered."""
        return key in self._items

    def register(self, key: TKey, item: TItem | Any) -> None:
        """Register an item."""
        if key in self._items:
            msg = f"Item already registered: {key}"
            raise self._error_class(msg)

        validated_item = self._validate_item(item)
        self._items[key] = validated_item

    def get(self, key: TKey) -> TItem:  # type: ignore
        """Get an item by key."""
        return self[key]

    def list_items(self) -> Sequence[TKey]:
        """List all registered item keys."""
        return list(self._items.keys())

    def reset(self) -> None:
        """Reset registry to initial state."""
        self._items.clear()
        self._configs.clear()
        self._initialized = False

    async def startup(self) -> None:
        """Initialize all registered items."""
        if self._initialized:
            return

        try:
            for item in self._items.values():
                await self._initialize_item(item)
            self._initialized = True
        except Exception as exc:
            await self.shutdown()
            msg = f"Registry startup failed: {exc}"
            raise self._error_class(msg) from exc

    async def shutdown(self) -> None:
        """Cleanup all registered items."""
        if not self._initialized:
            return

        errors: list[tuple[TKey, Exception]] = []

        for key, item in self._items.items():
            try:
                await self._cleanup_item(item)
            except Exception as exc:  # noqa: BLE001
                errors.append((key, exc))

        self._initialized = False

        if errors:
            error_msgs = [f"{key}: {exc}" for key, exc in errors]
            msg = f"Errors during shutdown: {', '.join(error_msgs)}"
            raise self._error_class(msg)

    @property
    @abstractmethod
    def _error_class(self) -> type[exceptions.LLMLingError]:
        """Error class to use for this registry."""

    @abstractmethod
    def _validate_item(self, item: Any) -> TItem:
        """Validate and possibly transform item before registration."""

    async def _initialize_item(self, item: TItem) -> None:
        """Initialize an item during startup."""
        if hasattr(item, "startup") and callable(item.startup):
            await item.startup()

    async def _cleanup_item(self, item: TItem) -> None:
        """Clean up an item during shutdown."""
        if hasattr(item, "shutdown") and callable(item.shutdown):
            await item.shutdown()

    # Implementing MutableMapping methods
    def __getitem__(self, key: TKey) -> TItem:
        try:
            return self._items[key]
        except KeyError as exc:
            msg = f"Item not found: {key}"
            raise self._error_class(msg) from exc

    def __setitem__(self, key: TKey, value: Any) -> None:
        """Support dict-style assignment."""
        self.register(key, value)

    def __contains__(self, key: object) -> bool:
        """Support 'in' operator without raising exceptions."""
        return key in self._items

    def __delitem__(self, key: TKey) -> None:
        if key in self._items:
            del self._items[key]
        else:
            msg = f"Item not found: {key}"
            raise self._error_class(msg)

    def __iter__(self) -> Iterator[TKey]:
        return iter(self._items)

    async def __aiter__(self):
        """Async iterate over items, ensuring they're initialized."""
        if not self._initialized:
            await self.startup()
        for key, item in self._items.items():
            yield key, item

    def __len__(self) -> int:
        return len(self._items)