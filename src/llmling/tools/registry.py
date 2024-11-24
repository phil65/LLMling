"""Registry for LLM-callable tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling.config.models import ToolConfig
from llmling.core.baseregistry import BaseRegistry
from llmling.core.log import get_logger
from llmling.tools.base import LLMCallableTool
from llmling.tools.exceptions import ToolError, ToolNotFoundError
from llmling.utils import importing


if TYPE_CHECKING:
    from types import ModuleType

    import py2openai

    from llmling.task.models import TaskContext, TaskProvider


logger = get_logger(__name__)


class ToolRegistry(BaseRegistry[str, LLMCallableTool]):
    """Registry for functions that can be called by LLMs."""

    @property
    def _error_class(self) -> type[ToolError]:
        """Error class to use for this registry."""
        return ToolError

    def _validate_item(self, item: Any) -> LLMCallableTool:
        """Validate and transform item into a LLMCallableTool."""
        match item:
            # Keep existing behavior for these cases
            case type() if issubclass(item, LLMCallableTool):
                return item()
            case LLMCallableTool():
                return item
            case ToolConfig():  # Handle Pydantic models
                return LLMCallableTool.from_callable(
                    item.import_path,
                    name_override=item.name,
                    description_override=item.description,
                )
            case dict() if "import_path" in item:  # Config dict
                return LLMCallableTool.from_callable(
                    item["import_path"],
                    name_override=item.get("name"),
                    description_override=item.get("description"),
                )
            case str():  # Import path
                return LLMCallableTool.from_callable(item)
            # Add new support for callables
            case _ if callable(item):
                return LLMCallableTool.from_callable(item)
            case _:
                msg = f"Invalid tool type: {type(item)}"
                raise ToolError(msg)

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

    def add_container(
        self,
        obj: type | ModuleType | Any,
        *,
        prefix: str = "",
        include_imported: bool = False,
    ) -> None:
        """Register all public callable members from a Python object.

        Args:
            obj: Any Python object to inspect (module, class, instance)
            prefix: Optional prefix for registered function names
            include_imported: Whether to include imported/inherited callables
        """
        for name, func in importing.get_pyobject_members(
            obj,
            include_imported=include_imported,
        ):
            self.register(f"{prefix}{name}", func)
            logger.debug("Registered callable %s as %s", name, f"{prefix}{name}")

    def get_schema(self, name: str) -> py2openai.OpenAIFunctionTool:
        """Get OpenAI function schema for a registered function.

        Args:
            name: Name of the registered function

        Returns:
            OpenAI function schema

        Raises:
            ToolError: If function not found
        """
        try:
            tool = self.get(name)
            return tool.get_schema()
        except KeyError as exc:
            msg = f"Function {name} not found"
            raise ToolError(msg) from exc

    def get_schemas(self) -> list[py2openai.OpenAIFunctionTool]:
        """Get schemas for all registered functions.

        Returns:
            List of OpenAI function schemas
        """
        return [self.get_schema(name) for name in self._items]

    async def execute(self, name: str, **params: Any) -> Any:
        """Execute a registered function.

        Args:
            name: Name of the function to execute
            **params: Parameters to pass to the function

        Returns:
            Function result

        Raises:
            ToolNotFoundError: If function not found
            ToolError: If execution fails
        """
        try:
            tool = self.get(name)
        except KeyError as exc:
            msg = f"Function {name} not found"
            raise ToolNotFoundError(msg) from exc

        # Let the original exception propagate
        return await tool.execute(**params)
