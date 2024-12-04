"""Watchdog-based file monitoring implementation."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import pathspec
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from llmling.core.log import get_logger
from llmling.monitors.files import FileEvent, FileEventType, FileMonitorCallback


if TYPE_CHECKING:
    import watchdog.observers.api

logger = get_logger(__name__)


class _WatchdogHandler(FileSystemEventHandler):
    """Handler for watchdog events.

    Converts watchdog events to our normalized FileEvents and handles debouncing.
    Only forwards events that match the configured patterns and pass debouncing.
    """

    def __init__(
        self,
        patterns: list[str] | None,
        callback: FileMonitorCallback,
        debounce_interval: float,
    ) -> None:
        """Initialize handler with patterns and callback.

        Args:
            patterns: File patterns to match (.gitignore style)
            callback: Function to call with file events
            debounce_interval: Minimum time between events
        """
        self.callback = callback
        self.spec = pathspec.PathSpec.from_lines(
            pathspec.patterns.GitWildMatchPattern,
            patterns or ["*"],
        )
        self._last_event_time = 0.0
        self._debounce_interval = debounce_interval

    def on_any_event(self, event: FileSystemEvent) -> None:
        """Handle any file system event.

        Args:
            event: Watchdog event to process
        """
        if event.is_directory:
            return

        # Debounce events
        current_time = time.time()
        if current_time - self._last_event_time < self._debounce_interval:
            return
        self._last_event_time = current_time

        # Convert path for pattern matching
        path = (
            event.src_path.decode()
            if isinstance(event.src_path, bytes)
            else event.src_path
        )
        rel_path = os.path.relpath(path, start=os.path.dirname(path))  # noqa: PTH120

        if not self.spec.match_file(rel_path):
            return

        # Map watchdog events to our event types
        event_type = {
            "created": FileEventType.ADDED,
            "modified": FileEventType.MODIFIED,
            "deleted": FileEventType.DELETED,
            "moved": FileEventType.MOVED,
        }.get(event.event_type, FileEventType.MODIFIED)

        self.callback([
            FileEvent(
                event_type=event_type,
                path=path,
                is_directory=event.is_directory,
            )
        ])


class WatchdogMonitor:
    """Watchdog-based file monitor implementation.

    Uses watchdog's Observer to watch files and directories.
    Handles multiple watched paths with individual handlers and pattern matching.
    """

    def __init__(self, *, debounce_interval: float = 0.1) -> None:
        """Initialize monitor.

        Args:
            debounce_interval: Minimum time between events in seconds
        """
        self._observer: watchdog.observers.api.BaseObserver | None = None
        self._handlers: dict[str, _WatchdogHandler] = {}
        self._watched_paths: set[str] = set()
        self._debounce_interval = debounce_interval

    async def start(self) -> None:
        """Start the file monitor."""
        if self._observer:
            return

        try:
            self._observer = Observer()
            self._observer.start()
            logger.debug("File monitor started")
        except Exception:
            self._observer = None
            logger.exception("Failed to start file monitor")
            raise

    async def stop(self) -> None:
        """Stop the file monitor."""
        if not self._observer:
            return

        try:
            self._observer.stop()
            self._observer.join(timeout=2.0)
            logger.debug("File monitor stopped")
        except Exception:
            logger.exception("Error stopping file monitor")
            raise
        finally:
            self._observer = None
            self._handlers.clear()
            self._watched_paths.clear()

    def add_watch(
        self,
        path: str | os.PathLike[str],
        patterns: list[str] | None = None,
        callback: FileMonitorCallback | None = None,
    ) -> None:
        """Add a path to monitor.

        Args:
            path: Path to monitor
            patterns: Optional file patterns to match
            callback: Callback to invoke on changes

        Raises:
            RuntimeError: If monitor not started
            ValueError: If callback not provided
        """
        if not self._observer:
            msg = "Monitor not started"
            raise RuntimeError(msg)

        if not callback:
            msg = "Callback is required"
            raise ValueError(msg)

        path_str = str(path)
        try:
            # Create handler with patterns and debouncing
            handler = _WatchdogHandler(
                patterns,
                callback,
                self._debounce_interval,
            )
            self._handlers[path_str] = handler

            # Schedule directory watching
            watch_path = os.path.dirname(path_str)  # noqa: PTH120
            if watch_path not in self._watched_paths:
                self._observer.schedule(handler, watch_path, recursive=True)
                self._watched_paths.add(watch_path)
                logger.debug("Added watch for: %s", path_str)

        except Exception:
            logger.exception("Failed to add watch for: %s", path_str)
            raise

    def remove_watch(self, path: str | os.PathLike[str]) -> None:
        """Remove a watched path.

        Args:
            path: Path to stop monitoring
        """
        path_str = str(path)
        if _handler := self._handlers.pop(path_str, None):
            try:
                # Observer cleanup happens in stop()
                logger.debug("Removed watch for: %s", path_str)
            except Exception:
                logger.exception("Error removing watch for: %s", path_str)
