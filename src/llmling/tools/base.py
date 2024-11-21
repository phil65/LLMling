from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import py2openai
from pydantic import BaseModel, ConfigDict, Field

from llmling.core.baseregistry import BaseRegistry
from llmling.tools.exceptions import ToolError
from llmling.utils import calling


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from llmling.tools import exceptions


class ToolSchema(BaseModel):
    """OpenAPI-compatible schema for a tool."""

    # Make type a required field with a default value
    type: Literal["function"] = Field(
        default="function",
        # Ensure it's always included in output
        exclude=False,
    )
    function: dict[str, Any]

    model_config = ConfigDict(
        frozen=True,
        # Ensure type is always included even when using exclude_unset
        populate_by_name=True,
    )


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


class ToolRegistry(BaseRegistry[BaseTool | DynamicTool, str]):
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
            case _:
                msg = f"Invalid tool type: {type(item)}"
                raise ToolError(msg)

    # Backward compatibility methods
    def register(self, tool: type[BaseTool] | BaseTool) -> None:  # type: ignore
        """Register a tool class or instance."""
        super().register(tool.name, tool)

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
        super().register(tool.name, tool)

    def get_schema(self, name: str) -> ToolSchema:
        """Get schema for a tool."""
        tool = self.get(name)
        return tool.get_schema()

    async def execute(self, name: str, **params: Any) -> Any:
        """Execute a tool by name."""
        logger.debug("Attempting to execute tool: %s", name)
        tool = self.get(name)
        return await tool.execute(**params)
