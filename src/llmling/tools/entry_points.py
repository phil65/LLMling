"""Entry point based toolset implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llmling.core.log import get_logger
from llmling.tools.toolsets import ToolSet


logger = get_logger(__name__)


class EntryPointTools(ToolSet):
    """Collection of tools from entry points."""

    def __init__(self, module: str) -> None:
        """Initialize entry point tools.

        Args:
            module: Module name to load tools from
        """
        from epregistry import EntryPointRegistry

        self.module = module
        self._tools: list[Callable[..., Any]] = []
        self.registry = EntryPointRegistry[Callable[..., Any]]("llmling")
        self._load_tools()

    def _load_tools(self) -> None:
        """Load tools from entry points."""
        entry_point = self.registry.get("tools")
        if not entry_point:
            msg = f"No tools entry point found for {self.module}"
            raise ValueError(msg)
        get_tools = entry_point.load()
        for item in get_tools():
            try:
                self._tools.append(item)
                name = getattr(item, "__name__", str(item))
                logger.debug("Loaded tool %s from entry point %s", name, self.module)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load tool from %s: %s", self.module, exc)

    def get_tools(self) -> list[Callable[..., Any]]:
        """Get all tools loaded from entry points."""
        return self._tools
