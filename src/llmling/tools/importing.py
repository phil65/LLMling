"""Toolset importing utilities."""

from __future__ import annotations

from typing import Any

from llmling.core.log import get_logger
from llmling.tools.toolsets import ToolSet
from llmling.utils import importing


logger = get_logger(__name__)


def import_toolset(import_path: str, **kwargs: Any) -> ToolSet:
    """Import and instantiate a ToolSet class.

    Args:
        import_path: Dotted path to the ToolSet class
        **kwargs: Arguments to pass to the ToolSet constructor

    Returns:
        Instantiated ToolSet
    """
    try:
        cls = importing.import_class(import_path)
        if not issubclass(cls, ToolSet):
            msg = f"{import_path} must be a ToolSet class"
            raise TypeError(msg)  # noqa: TRY301
        return cls(**kwargs)
    except Exception as exc:
        msg = f"Failed to import ToolSet from {import_path}"
        raise ValueError(msg) from exc
