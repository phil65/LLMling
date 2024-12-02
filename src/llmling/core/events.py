"""Event types for registry observers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from llmling.core.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable

    from llmling.core.typedefs import MetadataDict

logger = get_logger(__name__)


TKey = TypeVar("TKey")
TItem = TypeVar("TItem")


class EventType(Enum):
    """Registry state change events."""

    # Resource events
    RESOURCE_ADDED = auto()
    RESOURCE_MODIFIED = auto()
    RESOURCE_REMOVED = auto()
    RESOURCE_LIST_CHANGED = auto()

    # Tool events
    TOOL_ADDED = auto()
    TOOL_MODIFIED = auto()
    TOOL_REMOVED = auto()
    TOOL_LIST_CHANGED = auto()

    # Prompt events
    PROMPT_ADDED = auto()
    PROMPT_MODIFIED = auto()
    PROMPT_REMOVED = auto()
    PROMPT_LIST_CHANGED = auto()

    # General registry events
    REGISTRY_RESET = auto()


@dataclass
class Event:
    """Unified event type for LLMling."""

    type: EventType
    source: str  # Component that emitted the event
    timestamp: datetime = field(default_factory=datetime.now)
    name: str | None = None  # Item name if applicable
    data: Any | None = None  # The actual item/data
    metadata: dict[str, Any] | None = field(default_factory=dict)

    def __repr__(self) -> str:
        """Readable string representation."""
        return (
            f"Event(type={self.type.name}, source={self.source!r}, "
            f"name={self.name!r}, timestamp={self.timestamp})"
        )


class EventHandler(Protocol):
    """Protocol for unified event handling."""

    async def handle_event(self, event: Event) -> None:
        """Handle an event."""


class EventEmitter:
    """Base class for components that emit events."""

    def __init__(self) -> None:
        """Initialize event emitter."""
        self._event_handlers: set[EventHandler] = set()

    async def emit_event(self, event: Event) -> None:
        """Emit an event to all registered handlers."""
        for handler in self._event_handlers:
            try:
                await handler.handle_event(event)
            except Exception:
                logger.exception("Error handling event: %s", event)

    def add_event_handler(self, handler: EventHandler) -> None:
        """Add an event handler."""
        self._event_handlers.add(handler)

    def remove_event_handler(self, handler: EventHandler) -> None:
        """Remove an event handler."""
        self._event_handlers.discard(handler)


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
