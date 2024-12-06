"""Core interfaces for file monitoring."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    import os


class FileEventType(Enum):
    """Type of file system event."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass(frozen=True, slots=True)
class FileEvent:
    """Normalized file system event.

    Represents a file system change event in a normalized format,
    independent of the underlying file watching implementation.
    """

    event_type: FileEventType
    """Type of event (added/modified/deleted/moved)."""

    path: str
    """Path that triggered the event."""

    is_directory: bool = False
    """Whether the event was triggered by a directory."""


FileMonitorCallback = Callable[[list[FileEvent]], None]


class FileMonitor(Protocol):
    """Abstract interface for file monitoring.

    Defines the contract for file monitoring implementations.
    The monitoring flow is:
    1. Monitor is started via start()
    2. Paths are registered via add_watch()
    3. File changes trigger callbacks with FileEvent lists
    4. Monitoring is stopped via stop()
    """

    def __init__(self, *, debounce_interval: float = 0.1) -> None:
        """Initialize monitor with configuration.

        Args:
            debounce_interval: Minimum time between events in seconds
        """

    async def start(self) -> None:
        """Start monitoring."""

    async def stop(self) -> None:
        """Stop monitoring."""

    def add_watch(
        self,
        path: str | os.PathLike[str],
        patterns: list[str] | None = None,
        callback: FileMonitorCallback | None = None,
    ) -> None:
        """Add a path to monitor.

        Args:
            path: Path to monitor
            patterns: Optional file patterns to match (.gitignore style)
            callback: Callback to invoke on changes
        """

    def remove_watch(self, path: str | os.PathLike[str]) -> None:
        """Remove a watched path.

        Args:
            path: Path to stop monitoring
        """
