"""Watchfiles-based file monitoring implementation."""

from __future__ import annotations

import asyncio
import fnmatch
import pathlib
from typing import TYPE_CHECKING

from watchfiles import Change, awatch

from llmling.core.log import get_logger
from llmling.monitors.files import FileEvent, FileEventType, FileMonitorCallback


if TYPE_CHECKING:
    import os


logger = get_logger(__name__)


class WatchfilesMonitor:
    """Watchfiles-based file monitor implementation."""

    def __init__(
        self,
        *,
        debounce_ms: int = 1600,
        step_ms: int = 50,
        polling: bool | None = None,
        poll_delay_ms: int = 300,
    ) -> None:
        """Initialize monitor.

        Args:
            debounce_ms: Time to wait for collecting changes (milliseconds)
            step_ms: Time between checks (milliseconds)
            polling: Whether to force polling mode (None = auto)
            poll_delay_ms: Delay between polls if polling is used
        """
        self._running = False
        self._watches: dict[str, tuple[set[str], FileMonitorCallback]] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._debounce_ms = debounce_ms
        self._step_ms = step_ms
        self._polling = polling
        self._poll_delay_ms = poll_delay_ms

    async def start(self) -> None:
        self._running = True
        logger.debug("File monitor started")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._watches.clear()
        logger.debug("File monitor stopped")

    def add_watch(
        self,
        path: str | os.PathLike[str],
        patterns: list[str] | None = None,
        callback: FileMonitorCallback | None = None,
    ) -> None:
        if not self._running:
            msg = "Monitor not started"
            raise RuntimeError(msg)

        if not callback:
            msg = "Callback is required"
            raise ValueError(msg)

        path_str = str(path)

        # Store patterns as set for faster lookup
        watch_patterns = set(patterns) if patterns else set()
        self._watches[path_str] = (watch_patterns, callback)

        # Create watch task
        coro = self._watch_path(path_str)
        task = asyncio.create_task(coro, name=f"watch-{path_str}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        logger.debug("Added watch for: %s", path_str)

    def remove_watch(self, path: str | os.PathLike[str]) -> None:
        path_str = str(path)
        self._watches.pop(path_str, None)
        logger.debug("Removed watch for: %s", path_str)

    async def _watch_path(self, path: str) -> None:
        """Watch a path using watchfiles."""
        try:
            patterns, callback = self._watches[path]

            def should_notify(change_type: Change, changed_path: str) -> bool:
                if patterns:
                    return any(
                        fnmatch.fnmatch(pathlib.Path(changed_path).name, pattern)
                        for pattern in patterns
                    )
                return True

            logger.debug("Starting watch on %s with patterns %s", path, patterns)

            async for changes in awatch(
                path,
                watch_filter=should_notify,
                debounce=self._debounce_ms,
                step=self._step_ms,
                force_polling=self._polling,
                poll_delay_ms=self._poll_delay_ms,
                recursive=True,
                yield_on_timeout=False,
            ):
                if not self._running:
                    break

                if changes:
                    logger.debug("Changes detected: %s", changes)
                    events = [
                        FileEvent(
                            event_type=self._convert_event_type(change_type),
                            path=changed_path,
                            is_directory=pathlib.Path(changed_path).is_dir(),
                        )
                        for change_type, changed_path in changes
                    ]
                    if events:
                        logger.debug("Notifying callback with events: %s", events)
                        callback(events)

        except asyncio.CancelledError:
            logger.debug("Watch cancelled for: %s", path)
        except Exception:
            logger.exception("Watch error for: %s", path)

    def _convert_event_type(self, change: Change) -> FileEventType:
        return {
            Change.added: FileEventType.ADDED,
            Change.modified: FileEventType.MODIFIED,
            Change.deleted: FileEventType.DELETED,
        }.get(change, FileEventType.MODIFIED)
