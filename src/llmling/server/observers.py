"""Observer implementations for converting registry events to MCP notifications."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import logfire

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
    """Base observer with server reference and task tracking."""

    def __init__(self, server: LLMLingServer) -> None:
        """Initialize with server reference.

        Args:
            server: Server instance to notify
        """
        self.server = server

    def _create_notification_task(self, coro: Coroutine[None, None, Any]) -> None:
        """Create a tracked notification task.

        Args:
            coro: Coroutine to run as notification task
        """
        self.server._create_task(coro)

    async def cleanup(self) -> None:
        """Clean up observer resources."""


@logfire.instrument("Resource observer notification")
class ResourceObserver(ServerObserver):
    """Converts resource registry events to MCP notifications."""

    def __init__(self, server: LLMLingServer) -> None:
        """Initialize and create events object.

        Args:
            server: Server instance to notify
        """
        super().__init__(server)
        self.events = RegistryEvents[str, Resource]()
        # Set up callbacks
        self.events.on_item_added = self._handle_resource_changed
        self.events.on_item_modified = self._handle_resource_changed
        self.events.on_item_removed = self._handle_list_changed
        self.events.on_reset = self._handle_list_changed

    def _handle_resource_changed(self, key: str, resource: Resource) -> None:
        """Handle individual resource changes.

        Args:
            key: Resource name
            resource: Modified resource
        """
        try:
            loader = self.server.runtime.get_resource_loader(resource)
            uri = loader.create_uri(name=key)
            self._create_notification_task(self.server.notify_resource_change(uri))
        except Exception:
            logger.exception("Failed to notify resource change for %s", key)

    def _handle_list_changed(self, *args: object) -> None:
        """Handle changes that affect the resource list."""
        try:
            self._create_notification_task(self.server.notify_resource_list_changed())
        except Exception:
            logger.exception("Failed to notify resource list change")


@logfire.instrument("Prompt observer notification")
class PromptObserver(ServerObserver):
    """Converts prompt registry events to MCP notifications."""

    def __init__(self, server: LLMLingServer) -> None:
        """Initialize and create events object.

        Args:
            server: Server instance to notify
        """
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
            self._create_notification_task(self.server.notify_prompt_list_changed())
        except Exception:
            logger.exception("Failed to notify prompt list change")


@logfire.instrument("Tool observer notification")
class ToolObserver(ServerObserver):
    """Converts tool registry events to MCP notifications."""

    def __init__(self, server: LLMLingServer) -> None:
        """Initialize and create events object.

        Args:
            server: Server instance to notify
        """
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
            self._create_notification_task(self.server.notify_tool_list_changed())
        except Exception:
            logger.exception("Failed to notify tool list change")
