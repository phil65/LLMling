"""Registry for LLM-callable tools."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

import logfire

from llmling.core.baseregistry import BaseRegistry
from llmling.core.log import get_logger
from llmling.tools.base import LLMCallableTool
from llmling.tools.exceptions import ToolError, ToolNotFoundError
from llmling.utils import importing


if TYPE_CHECKING:
    from types import ModuleType

    import py2openai


logger = get_logger(__name__)


class ToolRegistry(BaseRegistry[str, LLMCallableTool]):
    """Registry for functions that can be called by LLMs."""

    @property
    def _error_class(self) -> type[ToolError]:
        """Error class to use for this registry."""
        return ToolError

    def _validate_item(self, item: Any) -> LLMCallableTool:  # noqa: PLR0911
        """Validate and transform item into a LLMCallableTool."""
        from llmling.config.models import ToolConfig

        match item:
            case LLMCallableTool():
                return item
            case ToolConfig():  # Handle Pydantic models
                obj_class = importing.import_class(item.import_path)

                # Check for crewai tools
                if importlib.util.find_spec("crewai"):
                    from crewai.tools import BaseTool as CrewAiBaseTool

                    if issubclass(obj_class, CrewAiBaseTool):
                        return LLMCallableTool.from_crewai_tool(
                            obj_class(
                                name=obj_class.__name__,
                                description=obj_class.description,  # type: ignore
                            ),
                            name_override=item.name,
                            description_override=item.description,
                        )

                # Check for langchain tools
                if importlib.util.find_spec("langchain_core"):
                    from langchain_core.tools import BaseTool as LangChainBaseTool

                    if issubclass(obj_class, LangChainBaseTool):
                        return LLMCallableTool.from_langchain_tool(
                            obj_class(),
                            name_override=item.name,
                            description_override=item.description,
                        )

                # Regular callable
                return LLMCallableTool.from_callable(
                    item.import_path,
                    name_override=item.name,
                    description_override=item.description,
                )
                # Either it wasn't a class or wasn't a crewai tool - treat as callable
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
                # Check for crewai tool instances
                if importlib.util.find_spec("crewai"):
                    from crewai.tools import BaseTool as CrewAiBaseTool

                    if isinstance(item, CrewAiBaseTool):
                        return LLMCallableTool.from_crewai_tool(item)

                # Check for langchain tool instances
                if importlib.util.find_spec("langchain_core"):
                    from langchain_core.tools import BaseTool as LangChainBaseTool

                    if isinstance(item, LangChainBaseTool):
                        return LLMCallableTool.from_langchain_tool(item)

                msg = f"Invalid tool type: {type(item)}"
                raise ToolError(msg)

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

    @logfire.instrument("Executing tool {_name}")
    async def execute(self, _name: str, **params: Any) -> Any:
        """Execute a registered function.

        Args:
            _name: Name of the function to execute
            **params: Parameters to pass to the function

        Returns:
            Function result

        Raises:
            ToolNotFoundError: If function not found
            ToolError: If execution fails
        """
        try:
            tool = self.get(_name)
        except KeyError as exc:
            msg = f"Function {_name} not found"
            raise ToolNotFoundError(msg) from exc

        # Let the original exception propagate
        return await tool.execute(**params)
