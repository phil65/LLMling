"""Utility functions for the watcher."""

from __future__ import annotations

from typing import TYPE_CHECKING

import upath


if TYPE_CHECKING:
    import os


def is_watchable_path(path: str | os.PathLike[str]) -> bool:
    """Check if a path can be watched.

    Args:
        path: Path to check

    Returns:
        True if path is local and can be watched
    """
    # return str(path).startswith(("/", "./", "../")) or ":" in str(path)
    return upath.UPath(path).protocol in ("file", "")
