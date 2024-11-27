from __future__ import annotations

from typing import Any

import pytest

from llmling.core import exceptions
from llmling.core.baseregistry import BaseRegistry
from llmling.core.events import RegistryEvents


# Simple test registry
class RegItem:
    """Test item for registry."""

    def __init__(self, value: str):
        self.value = value


class Registry(BaseRegistry[str, RegItem]):
    """Test registry implementation."""

    @property
    def _error_class(self) -> type[exceptions.LLMLingError]:
        return exceptions.LLMLingError

    def _validate_item(self, item: Any) -> RegItem:
        if isinstance(item, RegItem):
            return item
        if isinstance(item, str):
            return RegItem(item)
        msg = f"Invalid item type: {type(item)}"
        raise self._error_class(msg)


# Tests
def test_registry_observer_registration():
    """Test that observers can be registered and unregistered."""
    registry = Registry()
    events = RegistryEvents[str, RegItem]()

    # Register
    registry.add_observer(events)
    assert events in registry._observers

    # Unregister
    registry.remove_observer(events)
    assert events not in registry._observers


def test_item_added_event():
    """Test that item addition triggers event."""
    registry = Registry()
    events = RegistryEvents[str, RegItem]()

    # Track calls
    called_with = None

    def on_added(key: str, item: RegItem) -> None:
        nonlocal called_with
        called_with = (key, item)

    # Set up observer
    events.on_item_added = on_added
    registry.add_observer(events)

    # Add item
    test_item = RegItem("test")
    registry.register("test", test_item)

    # Check event was triggered
    assert called_with is not None
    key, item = called_with
    assert key == "test"
    assert item is test_item


def test_item_modified_event():
    """Test that item modification triggers event."""
    registry = Registry()
    events = RegistryEvents[str, RegItem]()

    # Track calls
    called_with = None

    def on_modified(key: str, item: RegItem) -> None:
        nonlocal called_with
        called_with = (key, item)

    # Set up observer
    events.on_item_modified = on_modified
    registry.add_observer(events)

    # Add and modify item
    registry.register("test", RegItem("original"))
    new_item = RegItem("modified")
    registry.register("test", new_item, replace=True)

    # Check event was triggered
    assert called_with is not None
    key, item = called_with
    assert key == "test"
    assert item is new_item


def test_item_removed_event():
    """Test that item removal triggers event."""
    registry = Registry()
    events = RegistryEvents[str, RegItem]()

    # Track calls
    called_with = None

    def on_removed(key: str, item: RegItem) -> None:
        nonlocal called_with
        called_with = (key, item)

    # Set up observer
    events.on_item_removed = on_removed
    registry.add_observer(events)

    # Add and remove item
    test_item = RegItem("test")
    registry.register("test", test_item)
    del registry["test"]

    # Check event was triggered
    assert called_with is not None
    key, item = called_with
    assert key == "test"
    assert item is test_item


def test_reset_event():
    """Test that reset triggers event."""
    registry = Registry()
    events = RegistryEvents[str, RegItem]()

    # Track calls
    was_called = False

    def on_reset() -> None:
        nonlocal was_called
        was_called = True

    # Set up observer
    events.on_reset = on_reset
    registry.add_observer(events)

    # Add item and reset
    registry.register("test", RegItem("test"))
    registry.reset()

    # Check event was triggered
    assert was_called


def test_multiple_observers():
    """Test that multiple observers receive events."""
    registry = Registry()
    events1 = RegistryEvents[str, RegItem]()
    events2 = RegistryEvents[str, RegItem]()

    # Track calls
    calls = []

    def on_added1(key: str, item: RegItem) -> None:
        calls.append(1)

    def on_added2(key: str, item: RegItem) -> None:
        calls.append(2)

    # Set up observers
    events1.on_item_added = on_added1
    events2.on_item_added = on_added2
    registry.add_observer(events1)
    registry.add_observer(events2)

    # Add item
    registry.register("test", RegItem("test"))

    # Check both observers were called
    assert len(calls) == 2  # noqa: PLR2004
    assert 1 in calls
    assert 2 in calls  # noqa: PLR2004


if __name__ == "__main__":
    pytest.main(["-vv", __file__])
