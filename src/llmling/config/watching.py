"""Config file watching functionality."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from llmling.core.log import get_logger
from llmling.monitors.implementations.watchfiles_watcher import WatchfilesMonitor


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    import os

    from llmling.monitors.files import FileEvent

logger = get_logger(__name__)


class ConfigWatcher:
    """Watches a config file for changes."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        callback: Callable[[], Awaitable[None]],
    ) -> None:
        """Initialize watcher for a config file.

        Args:
            path: Path to config file to watch
            callback: Async callback to call when file changes
        """
        self.path = str(path)
        self.callback = callback
        self._monitor = WatchfilesMonitor()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start watching the config file."""
        await self._monitor.start()

        def on_change(events: list[FileEvent]) -> None:
            """Handle file changes."""
            if not self._task or self._task.done():

                async def run_callback() -> None:
                    await self.callback()

                self._task = asyncio.create_task(run_callback())

        self._monitor.add_watch(
            self.path,
            patterns=["*.yml", "*.yaml"],  # Config file patterns
            callback=on_change,
        )
        logger.debug("Started watching config file: %s", self.path)

    async def stop(self) -> None:
        """Stop watching the config file."""
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

        await self._monitor.stop()
        logger.debug("Stopped watching config file: %s", self.path)
