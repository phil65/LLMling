"""Utilities for file watching."""

from __future__ import annotations

import asyncio
from functools import wraps
from threading import Lock
from typing import TYPE_CHECKING, Any

from upath import UPath

from llmling.core.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    import os


logger = get_logger(__name__)


def debounce(wait: float) -> Callable[[Callable[..., Any]], Callable[..., None]]:
    """Create a debounced version of a function."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., None]:
        last_called = 0.0
        lock = Lock()

        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> None:
            nonlocal last_called

            with lock:
                # Get main event loop
                try:
                    loop = asyncio.get_event_loop()
                    # Use call_soon_threadsafe to schedule from watchdog thread
                    loop.call_soon_threadsafe(fn, *args, **kwargs)
                except Exception:
                    logger.exception("Failed to schedule notification")

        return wrapped

    return decorator


def load_patterns(
    patterns: Sequence[str] | None = None,
    ignore_file: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Load and combine watch patterns.

    Args:
        patterns: List of patterns from config
        ignore_file: Optional path to .gitignore style file

    Returns:
        Combined list of patterns
    """
    result: list[str] = []

    # Add configured patterns
    if patterns:
        result.extend(patterns)

    # Add patterns from file
    if ignore_file:
        try:
            path = UPath(ignore_file)
            if path.exists():
                # Filter empty lines and comments
                file_patterns = [
                    line.strip()
                    for line in path.read_text("utf-8").splitlines()
                    if line.strip() and not line.startswith("#")
                ]
                result.extend(file_patterns)
        except Exception:
            logger.exception("Failed to load patterns from: %s", ignore_file)

    return result
