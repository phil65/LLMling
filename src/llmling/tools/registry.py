from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from llmling.config.models import ToolConfig
from llmling.core.baseregistry import BaseRegistry
from llmling.tools.base import BaseTool, DynamicTool, ToolSchema
from llmling.tools.exceptions import ToolError


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llmling.tools import exceptions


class ToolRegistry(BaseRegistry[str, BaseTool | DynamicTool]):
    """Registry for available tools."""

    @property
    def _error_class(self) -> type[exceptions.ToolError]:
        return ToolError

    def _validate_item(self, item: Any) -> BaseTool | DynamicTool:
        """Validate and possibly transform item before registration."""
        match item:
            case type() as cls if issubclass(cls, BaseTool):
                return cls()
            case BaseTool() | DynamicTool():
                return item
            case str():  # Just an import path
                return DynamicTool(import_path=item)
            case dict() if "import_path" in item:
                return DynamicTool(
                    import_path=item["import_path"],
                    name=item.get("name"),
                    description=item.get("description"),
                )
            case ToolConfig():  # Add support for ToolConfig
                return DynamicTool(
                    import_path=item.import_path,
                    name=item.name,
                    description=item.description,
                )
            case _:
                msg = f"Invalid tool type: {type(item)}"
                raise ToolError(msg)

    def get_schema(self, name: str) -> ToolSchema:
        """Get schema for a tool."""
        tool = self.get(name)
        return tool.get_schema()

    async def execute(self, name: str, **params: Any) -> Any:
        """Execute a tool by name."""
        logger.debug("Attempting to execute tool: %s", name)
        tool = self.get(name)
        return await tool.execute(**params)
