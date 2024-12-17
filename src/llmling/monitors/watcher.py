"""File monitoring using signals."""

from __future__ import annotations

import asyncio
import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

import psygnal
from watchfiles import Change, awatch

from llmling.core.log import get_logger


if TYPE_CHECKING:
    import os


logger = get_logger(__name__)


class FileWatcherSignals(psygnal.SignalGroup):
    """Signals for file system changes."""

    file_added = psygnal.Signal(str)  # path
    file_modified = psygnal.Signal(str)  # path
    file_deleted = psygnal.Signal(str)  # path
    watch_error = psygnal.Signal(str, Exception)  # path, error


class FileWatcher:
    """File system watcher using signals for change notification."""

    def __init__(
        self,
        *,
        debounce_ms: int = 1600,
        step_ms: int = 50,
        polling: bool | None = None,
        poll_delay_ms: int = 300,
    ) -> None:
        """Initialize watcher.

        Args:
            debounce_ms: Time to wait for collecting changes (milliseconds)
            step_ms: Time between checks (milliseconds)
            polling: Whether to force polling mode (None = auto)
            poll_delay_ms: Delay between polls if polling is used
        """
        self._running = False
        self._watches: dict[str, set[str]] = {}  # path -> patterns
        self._tasks: set[asyncio.Task[None]] = set()
        self._debounce_ms = debounce_ms
        self._step_ms = step_ms
        self._polling = polling
        self._poll_delay_ms = poll_delay_ms
        self.signals = FileWatcherSignals()

    async def start(self) -> None:
        self._running = True
        logger.debug("File watcher started")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._watches.clear()
        logger.debug("File watcher stopped")

    def add_watch(
        self,
        path: str | os.PathLike[str],
        patterns: list[str] | None = None,
    ) -> None:
        """Add a path to monitor."""
        if not self._running:
            msg = "Watcher not started"
            raise RuntimeError(msg)

        path_str = str(path)
        logger.debug("Setting up watch for %s with patterns %s", path_str, patterns)

        # Create watch task
        coro = self._watch_path(path_str, patterns or ["*"])
        task = asyncio.create_task(coro, name=f"watch-{path_str}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def remove_watch(self, path: str | os.PathLike[str]) -> None:
        path_str = str(path)
        self._watches.pop(path_str, None)
        logger.debug("Removed watch for: %s", path_str)

    async def _watch_path(self, path: str, patterns: list[str]) -> None:
        """Watch a path and emit signals for changes."""
        try:
            logger.debug("Starting watch on %s with patterns %s", path, patterns)

            async for changes in awatch(
                path,
                watch_filter=lambda _, p: any(
                    fnmatch.fnmatch(Path(p).name, pattern) for pattern in patterns
                ),
                debounce=self._debounce_ms,
                step=self._step_ms,
                recursive=True,
            ):
                if not self._running:
                    break

                for change_type, changed_path in changes:
                    logger.debug("Detected change: %s -> %s", change_type, changed_path)
                    match change_type:
                        case Change.added:
                            self.signals.file_added.emit(changed_path)
                        case Change.modified:
                            self.signals.file_modified.emit(changed_path)
                        case Change.deleted:
                            self.signals.file_deleted.emit(changed_path)
        except asyncio.CancelledError:
            logger.debug("Watch cancelled for: %s", path)
        except Exception as exc:
            logger.exception("Watch error for: %s", path)
            self.signals.watch_error.emit(path, exc)
