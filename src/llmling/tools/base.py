from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import TYPE_CHECKING, Any, ClassVar

import py2openai
from pydantic import BaseModel, ConfigDict

from llmling.tools.exceptions import ToolError
from llmling.utils import calling


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class ToolSchema(BaseModel):
    """OpenAPI-compatible schema for a tool."""

    type: str = "function"
    function: dict[str, Any]

    model_config = ConfigDict(frozen=True)


class BaseTool(ABC):
    """Base class for implementing complex tools that need state or custom logic."""

    # Class-level schema definition
    name: ClassVar[str]
    description: ClassVar[str]
    parameters_schema: ClassVar[dict[str, Any]]

    @classmethod
    def get_schema(cls) -> ToolSchema:
        """Get the tool's schema for LLM function calling."""
        schema = py2openai.create_schema(cls.execute).model_dump_openai()
        return ToolSchema(
            type="function",
            function={
                "name": cls.name,
                "description": cls.description,
                **schema,
            },
        )

    @abstractmethod
    async def execute(self, **params: Any) -> Any | Awaitable[Any]:
        """Execute the tool with given parameters."""


class DynamicTool:
    """Tool created from a function import path."""

    def __init__(
        self,
        import_path: str,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize tool from import path."""
        self.import_path = import_path
        self._func: Callable[..., Any] | None = None
        self._name = name
        self._description = description
        self._instance: BaseTool | None = None

    @property
    def name(self) -> str:
        """Get tool name."""
        if self._name:
            return self._name
        return self.import_path.split(".")[-1]

    @property
    def description(self) -> str:
        """Get tool description."""
        if self._description:
            return self._description
        if self.func.__doc__:
            return self.func.__doc__.strip()
        return f"Tool imported from {self.import_path}"

    @property
    def func(self) -> Callable[..., Any]:
        """Get the imported function."""
        if self._func is None:
            self._func = calling.import_callable(self.import_path)
        return self._func

    def get_schema(self) -> ToolSchema:
        """Generate schema from function signature."""
        func_schema = py2openai.create_schema(self.func)
        schema_dict = func_schema.model_dump_openai()

        # Override name and description
        schema_dict["name"] = self.name  # Use the tool's name, not the class name
        if self._description:
            schema_dict["description"] = self._description

        return ToolSchema(
            type="function",
            function=schema_dict,
        )

    async def execute(self, **params: Any) -> Any:
        """Execute the function."""
        if self._instance is None:
            # Import the class and create an instance
            cls = calling.import_callable(self.import_path)
            if isinstance(cls, type) and issubclass(cls, BaseTool):
                self._instance = cls()
                # Initialize the tool if needed
                if hasattr(self._instance, "startup"):
                    await self._instance.startup()
            else:
                # For regular functions, keep the old behavior
                return await calling.execute_callable(self.import_path, **params)

        # Execute using the tool instance
        return await self._instance.execute(**params)


class ToolRegistry:
    """Registry for available tools."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._tools: dict[str, BaseTool | DynamicTool] = {}

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def is_empty(self) -> bool:
        """Check if registry has any tools."""
        return not bool(self._tools)

    def register(self, tool: type[BaseTool] | BaseTool) -> None:
        """Register a tool class or instance."""
        if isinstance(tool, type):
            instance = tool()
            self._tools[tool.name] = instance
        else:
            self._tools[tool.name] = tool

    def register_path(
        self,
        import_path: str,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Register a tool from import path."""
        tool = DynamicTool(
            import_path=import_path,
            name=name,
            description=description,
        )
        if tool.name in self._tools:
            msg = f"Tool already registered: {tool.name}"
            raise ToolError(msg)
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> DynamicTool | BaseTool:
        """Get a tool by name."""
        try:
            return self._tools[name]
        except KeyError as exc:
            msg = f"Tool not found: {name}"
            raise ToolError(msg) from exc

    def get_schema(self, name: str) -> ToolSchema:
        """Get schema for a tool."""
        tool = self.get_tool(name)
        return tool.get_schema()

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    async def execute(self, name: str, **params: Any) -> Any:
        """Execute a tool by name."""
        logger.debug("Attempting to execute tool: %s", name)  # Add this
        tool = self.get_tool(name)
        return await tool.execute(**params)
