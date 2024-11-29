"""Extension loaders for different MCP component types."""

from __future__ import annotations

from llmling.extensions.base import BaseExtensionLoader
from llmling.tools.base import LLMCallableTool


class ToolsetLoader(BaseExtensionLoader[LLMCallableTool]):
    """Loads tools from entry points.

    Entry points should return a list of callable objects:

    def get_mcp_tools() -> list[Callable[..., Any]]:
        return [function1, function2]
    """

    component_type = "tools"
    converter = LLMCallableTool.from_callable
