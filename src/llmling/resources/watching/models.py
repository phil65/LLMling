"""Models for file watching configuration."""

from __future__ import annotations

from pydantic import BaseModel


class WatchConfig(BaseModel):
    """Watch configuration for resources."""

    enabled: bool = False
    """Whether the watch is enabled"""

    patterns: list[str] | None = None
    """List of pathspec patterns (.gitignore style)"""

    ignore_file: str | None = None
    """Path to .gitignore-style file"""
