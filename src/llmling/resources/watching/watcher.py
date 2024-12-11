"""File system watching for resources."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from llmling.core.log import get_logger
from llmling.monitors.implementations.watchfiles_watcher import WatchfilesMonitor
from llmling.monitors.utils import is_watchable_path
from llmling.resources.watching.utils import load_patterns


if TYPE_CHECKING:
    from llmling.config.models import BaseResource
    from llmling.monitors.files import FileEvent, FileMonitor
    from llmling.resources.registry import ResourceRegistry


logger = get_logger(__name__)


class ResourceWatcher:
    """Manages file system watching for resources.

    Coordinates file monitoring with resource management by:
    1. Setting up file monitoring for resources that request it
    2. Converting file changes to resource invalidations
    3. Managing monitor lifecycle

    The flow is:
    1. Resources are registered with watch configs
    2. File changes trigger callbacks
    3. Callbacks invalidate affected resources
    4. Registry handles reloading invalidated resources
    """

    def __init__(
        self,
        registry: ResourceRegistry,
        *,
        monitor: FileMonitor | None = None,
    ) -> None:
        """Initialize watcher.

        Args:
            registry: Registry to notify of changes
            monitor: Optional file monitor (uses WatchfilesMonitor by default)
        """
        self.registry = registry
        self.monitor = monitor or WatchfilesMonitor()
        self.handlers: dict[str, None] = {}  # For test compatibility
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        """Start the file system monitor.

        Raises:
            Exception: If monitor fails to start
        """
        try:
            self._loop = asyncio.get_running_loop()
            await self.monitor.start()
            logger.info("File system watcher started")
        except Exception:
            logger.exception("Failed to start file system watcher")
            raise

    async def stop(self) -> None:
        """Stop the file system monitor."""
        try:
            await self.monitor.stop()
            self.handlers.clear()
            self._loop = None
            logger.info("File system watcher stopped")
        except Exception:
            logger.exception("Error stopping file system watcher")

    def add_watch(self, name: str, resource: BaseResource) -> None:
        """Add a watch for a resource."""
        if not self._loop:
            msg = "Watcher not started"
            raise RuntimeError(msg)

        try:
            # Skip if resource doesn't provide a watch path
            if not (watch_path := resource.get_watch_path()):
                return

            patterns = load_patterns(
                patterns=resource.watch.patterns if resource.watch else None,
                ignore_file=None,
            )
            logger.debug("Setting up watch for %s with patterns: %s", name, patterns)

            def on_change(events: list[FileEvent]) -> None:
                """Handle file change events."""
                logger.debug("Received events for %s: %s", name, events)
                if self._loop and not self._loop.is_closed():
                    logger.debug("Invalidating resource: %s", name)
                    self._loop.call_soon_threadsafe(
                        self.registry.invalidate,
                        name,
                    )

            if is_watchable_path(watch_path):
                self.monitor.add_watch(watch_path, patterns=patterns, callback=on_change)
                self.handlers[name] = None
                logger.debug("Added watch for: %s -> %s", name, watch_path)
            else:
                logger.debug(
                    "Skipping watch for non-local path: %s -> %s",
                    name,
                    watch_path,
                )

        except Exception:
            logger.exception("Failed to add watch for: %s", name)

    def remove_watch(self, name: str) -> None:
        """Remove a watch for a resource.

        Args:
            name: Resource name to unwatch
        """
        try:
            self.handlers.pop(name, None)
            logger.debug("Removed watch for: %s", name)
        except Exception:
            logger.exception("Error removing watch for: %s", name)
