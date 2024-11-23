from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from llmling.config.models import ToolConfig
from llmling.core.baseregistry import BaseRegistry
from llmling.tools.base import LLMCallableTool
from llmling.tools.exceptions import ToolError


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from py2openai import OpenAIFunctionTool

    from llmling.tools import exceptions


class ToolRegistry(BaseRegistry[str, LLMCallableTool]):
    """Registry for available tools."""

    @property
    def _error_class(self) -> type[exceptions.ToolError]:
        return ToolError

    def _validate_item(self, item: Any) -> LLMCallableTool:
        """Validate and possibly transform item before registration."""
        match item:
            case type() as cls if issubclass(cls, LLMCallableTool):
                return cls()
            case LLMCallableTool():
                return item
            case str():  # Just an import path
                return LLMCallableTool.from_callable(item)
            case dict() if "import_path" in item:
                return LLMCallableTool.from_callable(
                    item["import_path"],
                    name_override=item.get("name"),
                    description_override=item.get("description"),
                )
            case ToolConfig():  # Add support for ToolConfig
                return LLMCallableTool.from_callable(
                    item.import_path,
                    name_override=item.name,
                    description_override=item.description,
                )
            case _:
                msg = f"Invalid tool type: {type(item)}"
                raise ToolError(msg)

    def get_schema(self, name: str) -> OpenAIFunctionTool:
        """Get schema for a tool."""
        tool = self.get(name)
        return tool.get_schema()

    async def execute(self, name: str, **params: Any) -> Any:
        """Execute a tool by name."""
        logger.debug("Attempting to execute tool: %s", name)
        tool = self.get(name)
        return await tool.execute(**params)
