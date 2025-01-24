"""File monitoring using signals."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import psygnal

from llmling.core.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
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

    signals = FileWatcherSignals()

    def __init__(
        self,
        *,
        debounce_ms: int = 1600,
        step_ms: int = 50,
        polling: bool | None = None,
        poll_delay_ms: int = 300,
        max_retries: int = 3,
    ) -> None:
        """Initialize watcher.

        Args:
            debounce_ms: Time to wait for collecting changes (milliseconds)
            step_ms: Time between checks (milliseconds)
            polling: Whether to force polling mode (None = auto)
            poll_delay_ms: Delay between polls if polling is used
            max_retries: Maximum number of retries for watch failures
        """
        self._running = False
        self._watches: dict[str, set[str]] = {}  # path -> patterns
        self._tasks: set[asyncio.Task[None]] = set()
        self._debounce_ms = debounce_ms
        self._step_ms = step_ms
        self._polling = polling
        self._poll_delay_ms = poll_delay_ms
        self._max_retries = max_retries

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
        # Validate path before creating watch task
        if not Path(path_str).exists():
            msg = f"Path does not exist: {path_str}"
            exc = FileNotFoundError(msg)
            self.signals.watch_error.emit(path_str, exc)
            return

        logger.debug("Setting up watch for %s with patterns %s", path_str, patterns)
        coro = self._watch_path(path_str, patterns or ["*"])
        task = asyncio.create_task(coro, name=f"watch-{path_str}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def remove_watch(self, path: str | os.PathLike[str]) -> None:
        path_str = str(path)
        self._watches.pop(path_str, None)
        logger.debug("Removed watch for: %s", path_str)

    @asynccontextmanager
    async def _watch_with_retries(self, path: str) -> AsyncIterator[None]:
        """Context manager for watching with retries."""
        retries = self._max_retries
        while retries and self._running:
            try:
                yield
                break  # Success - exit retry loop
            except asyncio.CancelledError:
                logger.debug("Watch cancelled for: %s", path)
                break
            except Exception as exc:
                msg = "Watch error for %s: %s. Retries left: %d"
                logger.warning(msg, path, exc, retries)
                retries -= 1
                if retries:
                    await asyncio.sleep(1)
                else:
                    logger.exception("Watch failed for: %s", path)
                    self.signals.watch_error.emit(path, exc)
                    break

    async def _watch_path(self, path: str, patterns: list[str]) -> None:
        """Watch a path and emit signals for changes."""
        import fnmatch

        from watchfiles import Change, awatch

        async with self._watch_with_retries(path):
            async for changes in awatch(
                path,
                watch_filter=lambda _, p: any(
                    fnmatch.fnmatch(Path(p).name, pattern) for pattern in patterns
                ),
                debounce=self._debounce_ms,
                step=self._step_ms,
                recursive=True,
                force_polling=self._polling,
                poll_delay_ms=self._poll_delay_ms,
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
