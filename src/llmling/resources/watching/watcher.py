"""File system watching implementation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pathspec
import upath
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from llmling.core.log import get_logger
from llmling.resources.watching.utils import debounce


if TYPE_CHECKING:
    from llmling.config.models import Resource
    from llmling.resources.registry import ResourceRegistry


logger = get_logger(__name__)


class ResourceEventHandler(FileSystemEventHandler):
    """Handles file system events for a specific resource."""

    def __init__(
        self,
        resource_name: str,
        registry: ResourceRegistry,
        patterns: list[str],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.resource_name = resource_name
        self.registry = registry
        self.loop = loop
        self.spec = pathspec.PathSpec.from_lines(
            pathspec.patterns.GitWildMatchPattern,
            patterns,
        )
        self._debounced_notify = debounce(wait=0.1)(self._notify_change)

    def on_any_event(self, event: FileSystemEvent) -> None:
        path = (
            event.src_path.decode()
            if isinstance(event.src_path, bytes)
            else event.src_path
        )
        if self.spec.match_file(path):
            # Use loop directly
            self.loop.call_soon_threadsafe(
                self.registry.invalidate,
                self.resource_name,
            )

    def _notify_change(self) -> None:
        """Notify registry of resource change."""
        try:
            self.registry.invalidate(self.resource_name)
        except Exception:
            logger.exception("Failed to notify change for %s", self.resource_name)


class ResourceWatcher:
    """Manages file system watching for resources."""

    def __init__(self, registry: ResourceRegistry) -> None:
        """Initialize watcher.

        Args:
            registry: Registry to notify of changes
        """
        self.registry = registry
        self.observer: Observer | None = None
        self.handlers: dict[str, ResourceEventHandler] = {}
        # Track watch directories to avoid duplicates
        self._watched_paths: set[str] = set()
        self.loop = asyncio.get_event_loop()

    async def start(self) -> None:
        """Start the file system observer."""
        if self.observer:
            return

        try:
            self.observer = Observer()
            self.observer.start()
            logger.info("File system watcher started")
        except Exception:
            self.observer = None
            logger.exception("Failed to start file system watcher")

    async def stop(self) -> None:
        """Stop the file system observer."""
        if not self.observer:
            return

        try:
            self.observer.stop()
            self.observer.join(timeout=2.0)
            self.observer = None
            self.handlers.clear()
            self._watched_paths.clear()
            logger.info("File system watcher stopped")
        except Exception:
            logger.exception("Error stopping file system watcher")

    def add_watch(self, name: str, resource: Resource) -> None:
        if not self.observer:
            return

        try:
            # Create handler with loop
            handler = ResourceEventHandler(
                name,
                self.registry,
                patterns=resource.watch.patterns or ["*"],  # type: ignore
                loop=self.loop,
            )
            self.handlers[name] = handler

            # Schedule directory watching
            if hasattr(resource, "path"):
                path = upath.UPath(resource.path).parent  # type: ignore
                if path not in self._watched_paths:
                    self.observer.schedule(handler, path, recursive=True)
                    self._watched_paths.add(str(path))
                    logger.debug("Added watch for: %s -> %s", name, path)

        except Exception:
            logger.exception("Failed to add watch for: %s", name)

    def remove_watch(self, name: str) -> None:
        """Remove a watch for a resource.

        Args:
            name: Resource name to unwatch
        """
        if _handler := self.handlers.pop(name, None):
            try:
                # Observer will be cleaned up on stop
                logger.debug("Removed watch for: %s", name)
            except Exception:
                logger.exception("Error removing watch for: %s", name)