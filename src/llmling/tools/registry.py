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

    from llmling.task.models import TaskContext, TaskProvider
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

    def get_tool_config(
        self,
        task_context: TaskContext,
        task_provider: TaskProvider,
    ) -> dict[str, Any] | None:
        """Get tool configuration for LLM parameters."""
        available_tools: list[str] = []

        # Add inherited tools from provider if enabled
        if (
            task_context.inherit_tools
            and task_provider.settings
            and task_provider.settings.tools
        ):
            logger.debug(
                "Inheriting tools from provider: %s", task_provider.settings.tools
            )
            available_tools.extend(task_provider.settings.tools)

        # Add task-specific tools
        if task_context.tools:
            logger.debug("Adding task-specific tools: %s", task_context.tools)
            available_tools.extend(task_context.tools)

        if not available_tools:
            logger.debug("No tools available")
            return None

        # Get complete tool schemas
        tool_schemas = [self.get_schema(name) for name in available_tools if name in self]

        if not tool_schemas:
            return None

        return {
            "tools": tool_schemas,
            "tool_choice": (
                task_context.tool_choice
                or (
                    task_provider.settings.tool_choice if task_provider.settings else None
                )
                or "auto"
            ),
        }
