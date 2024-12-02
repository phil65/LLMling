"""Tests for server observers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from llmling.config.models import TextResource
from llmling.server.observers import PromptObserver, ResourceObserver, ToolObserver


if TYPE_CHECKING:
    from llmling.core.events import Event


@pytest.fixture
def mock_server() -> Mock:
    """Create a mock server with required methods."""
    server = Mock()
    server._create_task = Mock(side_effect=asyncio.create_task)

    # Add async notification methods
    async def notify_change(uri: str) -> None: ...
    async def notify_list_changed() -> None: ...
    async def emit_event(event: Event) -> None: ...

    server.notify_resource_change = Mock(side_effect=notify_change)
    server.notify_resource_list_changed = Mock(side_effect=notify_list_changed)
    server.notify_prompt_list_changed = Mock(side_effect=notify_list_changed)
    server.notify_tool_list_changed = Mock(side_effect=notify_list_changed)

    # Mock runtime config with async emit_event
    mock_runtime = Mock()
    mock_runtime.get_resource_loader.return_value.create_uri.return_value = "test://uri"
    mock_runtime.emit_event = Mock(side_effect=emit_event)
    server.runtime = mock_runtime

    return server


@pytest.mark.asyncio
async def test_prompt_observer_notifications(mock_server: Mock) -> None:
    """Test that prompt observer triggers server notifications."""
    observer = PromptObserver(mock_server)

    observer._handle_prompt_list_changed()
    await asyncio.sleep(0)

    mock_server.notify_prompt_list_changed.assert_called_once()
    # Event emission + notification
    assert mock_server._create_task.call_count == 2  # noqa: PLR2004


@pytest.mark.asyncio
async def test_tool_observer_notifications(mock_server: Mock) -> None:
    """Test that tool observer triggers server notifications."""
    observer = ToolObserver(mock_server)

    observer._handle_tool_list_changed()
    await asyncio.sleep(0)

    mock_server.notify_tool_list_changed.assert_called_once()
    # Event emission + notification
    assert mock_server._create_task.call_count == 2  # noqa: PLR2004


@pytest.mark.asyncio
async def test_resource_observer_notifications(mock_server: Mock) -> None:
    """Test that resource observer triggers server notifications."""
    observer = ResourceObserver(mock_server)
    resource = TextResource(content="test")

    # Trigger events (each handler creates 2 tasks)
    observer._handle_resource_modified("test_key", resource)
    observer._handle_resource_list_changed()

    await asyncio.sleep(0)

    mock_server.notify_resource_change.assert_called_once_with("test://uri")
    mock_server.notify_resource_list_changed.assert_called_once()
    # Two handlers, two tasks each
    assert mock_server._create_task.call_count == 4  # noqa: PLR2004


@pytest.mark.asyncio
async def test_observer_error_handling(mock_server: Mock) -> None:
    """Test that observer handles server errors gracefully."""

    async def failing_notify(*args: object) -> None:
        msg = "Test error"
        raise RuntimeError(msg)

    mock_server.notify_resource_list_changed = Mock(side_effect=failing_notify)
    observer = ResourceObserver(mock_server)

    # Should not raise
    observer._handle_resource_list_changed()
    await asyncio.sleep(0)

    # Verify tasks were created despite error (one for event emission, one for notify)
    assert mock_server._create_task.call_count == 2  # noqa: PLR2004
