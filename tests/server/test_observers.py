from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from llmling.config.models import TextResource
from llmling.server.observers import PromptObserver, ResourceObserver, ToolObserver


@pytest.fixture
def mock_server() -> Mock:
    """Create a mock server with required methods."""
    server = Mock()

    # Add async notification methods
    async def notify_change(uri: str) -> None: ...
    async def notify_list_changed() -> None: ...

    server.notify_resource_change = Mock(side_effect=notify_change)
    server.notify_resource_list_changed = Mock(side_effect=notify_list_changed)
    server.notify_prompt_list_changed = Mock(side_effect=notify_list_changed)
    server.notify_tool_list_changed = Mock(side_effect=notify_list_changed)
    # Mock loader registry
    server.loader_registry.get_loader.return_value.create_uri.return_value = "test://uri"
    return server


@pytest.mark.asyncio
async def test_resource_observer_notifications(mock_server: Mock) -> None:
    """Test that resource observer creates notification tasks."""
    observer = ResourceObserver(mock_server)
    resource = TextResource(content="test")

    # Trigger events
    observer._handle_resource_changed("test_key", resource)
    observer._handle_list_changed()

    # Wait for tasks to complete
    await asyncio.sleep(0)

    # Check notifications were triggered
    mock_server.notify_resource_change.assert_called_once_with("test://uri")
    mock_server.notify_resource_list_changed.assert_called_once()


@pytest.mark.asyncio
async def test_prompt_observer_notifications(mock_server: Mock) -> None:
    """Test that prompt observer creates notification tasks."""
    observer = PromptObserver(mock_server)

    # Trigger event
    observer._handle_list_changed()

    # Wait for tasks to complete
    await asyncio.sleep(0)

    # Check notification was triggered
    mock_server.notify_prompt_list_changed.assert_called_once()


@pytest.mark.asyncio
async def test_tool_observer_notifications(mock_server: Mock) -> None:
    """Test that tool observer creates notification tasks."""
    observer = ToolObserver(mock_server)

    # Trigger event
    observer._handle_list_changed()

    # Wait for tasks to complete
    await asyncio.sleep(0)

    # Check notification was triggered
    mock_server.notify_tool_list_changed.assert_called_once()


@pytest.mark.asyncio
async def test_observer_task_cleanup() -> None:
    """Test that observer tasks are properly tracked and cleaned up."""
    server = Mock()

    # Make notification hang for a bit
    async def slow_notify(*args: object) -> None:
        await asyncio.sleep(0.1)

    server.notify_resource_list_changed = slow_notify

    observer = ResourceObserver(server)

    # Create a notification task
    observer._handle_list_changed()

    # Should have one task
    assert len(observer._tasks) == 1

    # Wait for task to complete
    await asyncio.sleep(0.2)

    # Task should be removed
    assert len(observer._tasks) == 0


@pytest.mark.asyncio
async def test_prompt_observer_task_cleanup() -> None:
    """Test that prompt observer tasks are properly tracked and cleaned up."""
    server = Mock()

    async def slow_notify(*args: object) -> None:
        await asyncio.sleep(0.1)

    server.notify_prompt_list_changed = slow_notify

    observer = PromptObserver(server)
    observer._handle_list_changed()
    assert len(observer._tasks) == 1
    await observer.cleanup()
    assert len(observer._tasks) == 0


@pytest.mark.asyncio
async def test_tool_observer_task_cleanup() -> None:
    """Test that tool observer tasks are properly tracked and cleaned up."""
    server = Mock()

    async def slow_notify(*args: object) -> None:
        await asyncio.sleep(0.1)

    server.notify_tool_list_changed = slow_notify

    observer = ToolObserver(server)
    observer._handle_list_changed()
    assert len(observer._tasks) == 1
    await observer.cleanup()
    assert len(observer._tasks) == 0
