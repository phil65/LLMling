"""Models for file watching configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WatchConfig(BaseModel):
    """Watch configuration for resources."""

    enabled: bool = False
    patterns: list[str] | None = Field(
        default=None,
        description="List of pathspec patterns (.gitignore style)",
    )
    ignore_file: str | None = Field(
        default=None,
        description="Path to .gitignore-style file",
    )
