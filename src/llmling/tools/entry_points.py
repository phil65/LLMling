"""Entry point based toolset implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from epregistry import EntryPointRegistry

from llmling.core.log import get_logger
from llmling.tools.toolsets import ToolSet


logger = get_logger(__name__)


class EntryPointTools(ToolSet):
    """Tool collection from entry points."""

    def __init__(self, module: str) -> None:
        """Initialize entry point tools.

        Args:
            module: Module name to load tools from
        """
        self.module = module
        self._tools: list[Callable[..., Any]] = []
        self._load_tools()

    def _load_tools(self) -> None:
        """Load tools from entry points."""
        try:
            registry = EntryPointRegistry[Callable[..., Any]]("llmling")
            if entry_point := registry.get("tools"):
                get_tools = entry_point.load()
                self._tools = get_tools()
                logger.debug(
                    "Loaded %d tools from entry point %s",
                    len(self._tools),
                    self.module,
                )
            else:
                msg = f"No tools entry point found for {self.module}"
                raise ValueError(msg)  # noqa: TRY301
        except Exception as exc:
            msg = f"Failed to load tools from {self.module}"
            raise ValueError(msg) from exc

    def get_tools(self) -> list[Callable[..., Any]]:
        """Get all tools loaded from entry points."""
        return self._tools
