"""Observer implementations for converting registry events to MCP notifications."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from llmling.config.models import Resource
from llmling.core.events import RegistryEvents
from llmling.core.log import get_logger
from llmling.prompts.models import Prompt
from llmling.tools.base import LLMCallableTool


if TYPE_CHECKING:
    from collections.abc import Coroutine

    from llmling.server.server import LLMLingServer


logger = get_logger(__name__)


class ServerObserver:
    """Base observer with server reference."""

    def __init__(self, server: LLMLingServer) -> None:
        """Initialize with server reference."""
        self.server = server
        self._tasks: set[asyncio.Task[None]] = set()

    def _create_notification_task(
        self,
        coro: Coroutine[None, None, None],
        name: str,
    ) -> None:
        """Create and track a notification task."""
        task: asyncio.Task[Any] = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


class ResourceObserver(ServerObserver):
    """Converts resource registry events to MCP notifications."""

    def __init__(self, server: LLMLingServer) -> None:
        """Initialize and create events object."""
        super().__init__(server)
        self.events = RegistryEvents[str, Resource]()
        # Set up callbacks
        self.events.on_item_added = self._handle_resource_changed
        self.events.on_item_modified = self._handle_resource_changed
        self.events.on_item_removed = self._handle_list_changed
        self.events.on_reset = self._handle_list_changed

    def _handle_resource_changed(self, key: str, resource: Resource) -> None:
        """Handle individual resource changes."""
        try:
            # Get appropriate loader to create URI
            loader = self.server.loader_registry.get_loader(resource)
            uri = loader.create_uri(name=key)
            # Create and track notification task
            self._create_notification_task(
                self.server.notify_resource_change(uri),
                f"notify_resource_change_{key}",
            )
        except Exception:
            logger.exception("Failed to notify resource change")

    def _handle_list_changed(self, *args: object) -> None:
        """Handle changes that affect the resource list."""
        try:
            self._create_notification_task(
                self.server.notify_resource_list_changed(),
                "notify_resource_list_changed",
            )
        except Exception:
            logger.exception("Failed to notify resource list change")


class PromptObserver(ServerObserver):
    """Converts prompt registry events to MCP notifications."""

    def __init__(self, server: LLMLingServer) -> None:
        """Initialize and create events object."""
        super().__init__(server)
        self.events = RegistryEvents[str, Prompt]()
        # Any prompt change triggers list update
        self.events.on_item_added = self._handle_list_changed
        self.events.on_item_modified = self._handle_list_changed
        self.events.on_item_removed = self._handle_list_changed
        self.events.on_reset = self._handle_list_changed

    def _handle_list_changed(self, *args: object) -> None:
        """Handle any prompt changes."""
        try:
            self._create_notification_task(
                self.server.notify_prompt_list_changed(),
                "notify_prompt_list_changed",
            )
        except Exception:
            logger.exception("Failed to notify prompt list change")


class ToolObserver(ServerObserver):
    """Converts tool registry events to MCP notifications."""

    def __init__(self, server: LLMLingServer) -> None:
        """Initialize and create events object."""
        super().__init__(server)
        self.events = RegistryEvents[str, LLMCallableTool]()
        # Any tool change triggers list update
        self.events.on_item_added = self._handle_list_changed
        self.events.on_item_modified = self._handle_list_changed
        self.events.on_item_removed = self._handle_list_changed
        self.events.on_reset = self._handle_list_changed

    def _handle_list_changed(self, *args: object) -> None:
        """Handle any tool changes."""
        try:
            self._create_notification_task(
                self.server.notify_tool_list_changed(),
                "notify_tool_list_changed",
            )
        except Exception:
            logger.exception("Failed to notify tool list change")
