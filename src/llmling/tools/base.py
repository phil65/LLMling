from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import TYPE_CHECKING, Any, ClassVar

import py2openai

from llmling.utils import calling


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class BaseTool(ABC):
    """Base class for implementing complex tools that need state or custom logic."""

    # Class-level schema definition
    name: ClassVar[str]
    description: ClassVar[str]
    parameters_schema: ClassVar[dict[str, Any]]

    @classmethod
    def get_schema(cls) -> py2openai.ToolSchema:
        """Get the tool's schema for LLM function calling."""
        return py2openai.create_schema(cls.execute).model_dump_openai()

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

    def get_schema(self) -> py2openai.ToolSchema:
        """Generate schema from function signature."""
        schema_dict = py2openai.create_schema(self.func).model_dump_openai()
        # Override name and description
        schema_dict["name"] = self.name or schema_dict["name"]
        schema_dict["description"] = self._description or schema_dict["description"]
        return schema_dict

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
