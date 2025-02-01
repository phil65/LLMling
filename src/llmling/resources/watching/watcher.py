"""File system watching for resources."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from llmling.core.log import get_logger
from llmling.resources.watching.utils import load_patterns
from llmling.utils.watcher import FileWatcher


if TYPE_CHECKING:
    from llmling.config.models import BaseResource
    from llmling.resources.registry import ResourceRegistry

logger = get_logger(__name__)


class ResourceWatcher:
    """Manages file system watching for resources."""

    def __init__(self, registry: ResourceRegistry) -> None:
        self.registry = registry
        self.watcher = FileWatcher()
        self._loop: asyncio.AbstractEventLoop | None = None

        # Debug log our signal connections
        logger.debug("Setting up file watcher signals")
        self.watcher.signals.file_modified.connect(self._on_file_changed)
        self.watcher.signals.watch_error.connect(self._on_watch_error)

    async def start(self) -> None:
        """Start the file system monitor."""
        try:
            self._loop = asyncio.get_running_loop()
            await self.watcher.start()
            logger.info("File system watcher started")
        except Exception:
            logger.exception("Failed to start file system watcher")
            raise

    async def stop(self) -> None:
        """Stop the file system monitor."""
        try:
            await self.watcher.stop()
            self._loop = None
            logger.info("File system watcher stopped")
        except Exception:
            logger.exception("Error stopping file system watcher")

    def add_watch(self, name: str, resource: BaseResource) -> None:
        """Add a watch for a resource."""
        import upath

        if not self._loop:
            msg = "Watcher not started"
            raise RuntimeError(msg)

        try:
            if not (watch_path := resource.get_watch_path()):
                return

            patterns = load_patterns(
                patterns=resource.watch.patterns if resource.watch else None,
                ignore_file=None,
            )
            # str(path).startswith(("/", "./", "../")) or ":" in str(path)
            if upath.UPath(watch_path).protocol in ("file", ""):
                self.watcher.add_watch(watch_path, patterns=patterns)
                logger.debug("Added watch for: %s -> %s", name, watch_path)
            else:
                msg = "Skipping watch for non-local path: %s -> %s"
                logger.debug(msg, name, watch_path)

        except Exception:
            logger.exception("Failed to add watch for: %s", name)

    def remove_watch(self, name: str) -> None:
        """Remove a watch for a resource."""
        try:
            # We only need resource name for logging now
            logger.debug("Removed watch for: %s", name)
        except Exception:
            logger.exception("Error removing watch for: %s", name)

    def _on_file_changed(self, path: str) -> None:
        """Handle file change events."""
        logger.debug("Received file change for: %s", path)
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._invalidate_resources_for_path, path)

    def _on_watch_error(self, path: str, exc: Exception) -> None:
        """Handle watch errors."""
        logger.error("Error watching path %s: %s", path, exc)

    def _invalidate_resources_for_path(self, path: str) -> None:
        """Invalidate any resources watching the given path."""
        path_obj = Path(path)
        for name, resource in self.registry.items():
            watch_path = Path(resource.get_watch_path() or "")
            logger.debug("Checking resource %s with watch path %s", name, watch_path)
            # If resource watches a directory, check if changed file is in it
            if watch_path.is_dir():
                if path_obj.is_relative_to(watch_path):
                    logger.debug("Invalidating resource %s (file in watched dir)", name)
                    self.registry.invalidate(name)
            # If resource watches a specific file, compare paths
            elif watch_path == path_obj:
                logger.debug("Invalidating resource %s (direct file match)", name)
                self.registry.invalidate(name)
