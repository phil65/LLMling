"""Event types for registry observers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable

    from llmling.core.typedefs import MetadataDict


TKey = TypeVar("TKey")
TItem = TypeVar("TItem")


class RegistryEvents(Generic[TKey, TItem]):
    """Event callbacks for registry changes."""

    def __init__(self) -> None:
        """Initialize empty callbacks."""
        self.on_item_added: Callable[[TKey, TItem], None] | None = None
        self.on_item_removed: Callable[[TKey, TItem], None] | None = None
        self.on_item_modified: Callable[[TKey, TItem], None] | None = None
        self.on_list_changed: Callable[[], None] | None = None
        self.on_reset: Callable[[], None] | None = None

    def __repr__(self) -> str:
        """Show which callbacks are set."""
        callbacks = []
        for name, cb in vars(self).items():
            if cb is not None:
                callbacks.append(name)
        return f"{self.__class__.__name__}(active={callbacks})"


class ResourceEvents(RegistryEvents[str, "Resource"]):  # type: ignore
    """Resource-specific registry events."""

    def __init__(self) -> None:
        """Initialize with resource-specific callbacks."""
        super().__init__()
        self.on_content_changed: Callable[[str, str | bytes], None] | None = None
        self.on_metadata_changed: Callable[[str, MetadataDict], None] | None = None
