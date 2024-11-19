from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, ClassVar

import py2openai
from pydantic import BaseModel, ConfigDict

from llmling.tools.exceptions import ToolError
from llmling.utils import calling


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
        return ToolSchema(
            type="function",
            function={
                "name": cls.name,
                "description": cls.description,
                "parameters": cls.parameters_schema,
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

        # Override description if custom one is provided
        if self._description:
            schema_dict["description"] = self._description

        return ToolSchema(
            type="function",
            function=schema_dict,
        )

    async def execute(self, **params: Any) -> Any:
        """Execute the function."""
        return await calling.execute_callable(self.import_path, **params)


class ToolRegistry(MutableMapping[str, BaseTool | DynamicTool]):
    """Registry for available tools."""

    def __init__(
        self,
        *args: Mapping[str, BaseTool | DynamicTool],
        **kwargs: BaseTool | DynamicTool,
    ) -> None:
        """Initialize registry with optional initial tools.

        Supports dict-like initialization:
        - ToolRegistry({'name': tool})
        - ToolRegistry(name=tool)
        - ToolRegistry([('name', tool)])
        """
        self._tools: dict[str, BaseTool | DynamicTool] = {}
        self.update(*args, **kwargs)

    def is_empty(self) -> bool:
        """Check if registry has any tools."""
        return not bool(self._tools)

    def register(self, tool: type[BaseTool] | BaseTool | DynamicTool) -> None:
        """Register a tool.

        Args:
            tool: Can be:
                - A BaseTool class (will be instantiated)
                - A BaseTool instance
                - A DynamicTool instance
        """
        match tool:
            case type() as tool_cls if issubclass(tool_cls, BaseTool):
                instance = tool_cls()
                self._tools[tool_cls.name] = instance
            case BaseTool() | DynamicTool() as instance:
                self._tools[instance.name] = instance
            case _:
                msg = f"Unsupported tool type: {type(tool)}"
                raise ToolError(msg)

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

    def get_schema(self, name: str) -> ToolSchema:
        """Get schema for a tool."""
        tool = self[name]
        return tool.get_schema()

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    async def execute(self, name: str, **params: Any) -> Any:
        """Execute a tool by name."""
        tool = self[name]
        return await tool.execute(**params)

    # Required abstract methods from MutableMapping
    def __getitem__(self, name: str) -> DynamicTool | BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            msg = f"Tool not found: {name}"
            raise ToolError(msg) from exc

    def __setitem__(
        self, key: str, value: type[BaseTool] | BaseTool | DynamicTool
    ) -> None:
        """Set a tool in the registry using dict-style assignment."""
        self.register(value)
        # If we need to override the name with the key:
        # if key != self._tools[value.name].name:
        #     self._tools[key] = self._tools.pop(value.name)

    def __delitem__(self, key: str) -> None:
        del self._tools[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)
