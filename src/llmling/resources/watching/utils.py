"""Utilities for file watching."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling.core.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence
    import os


logger = get_logger(__name__)


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
    from upathtools import to_upath

    result: list[str] = []

    # Add configured patterns
    if patterns:
        result.extend(patterns)

    # Add patterns from file
    if ignore_file:
        try:
            if (path := to_upath(ignore_file)).exists():
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
