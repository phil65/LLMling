from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import py2openai

from llmling.core.descriptors import classproperty
from llmling.core.log import get_logger
from llmling.utils import calling


T = TypeVar("T", bound="LLMCallableTool")

logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class LLMCallableTool(ABC):
    """Base class for implementing tools callable by an LLM via tool calling."""

    # Class-level schema definition
    name: ClassVar[str]
    description: ClassVar[str]
    _import_path: str | None = None  # For dynamic tools
    _original_callable: Callable[..., Any] | None = None  # For dynamic tools

    def __repr__(self) -> str:
        """Show tool name and import path."""
        return f"{self.__class__.__name__}(name={self.name!r}, path={self.import_path!r})"

    @property
    def system_prompt(self) -> str:
        """Get the system prompt for this tool.

        Override this to provide tool-specific instructions.
        """
        return ""

    @classproperty  # type: ignore
    def import_path(cls) -> str:  # noqa: N805
        """Get the import path of the tool.

        For class-based tools, returns the actual class import path.
        For dynamic tools, returns the stored import path.
        """
        if cls._import_path is not None:
            return cls._import_path
        # For class-based tools, get the actual class import path
        return f"{cls.__module__}.{cls.__qualname__}"  # type: ignore

    @classmethod
    def get_schema(cls) -> py2openai.OpenAIFunctionTool:
        """Get the tool's schema for LLM function calling."""
        # For dynamic tools, we want to use the original callable's schema
        if cls._original_callable:
            schema = py2openai.create_schema(cls._original_callable).model_dump_openai()
            schema["function"]["name"] = cls.name
            schema["function"]["description"] = cls.description
            return schema

        # For regular tools, use the execute method
        schema = py2openai.create_schema(cls.execute).model_dump_openai()
        schema["function"]["name"] = cls.name
        return schema

    @abstractmethod
    async def execute(self, **params: Any) -> Any | Awaitable[Any]:
        """Execute the tool with given parameters."""

    @classmethod
    def from_callable(
        cls,
        fn: Callable[..., Any] | str,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
    ) -> LLMCallableTool:
        """Create a tool instance from a callable or import path.

        Args:
            fn: Function or import path to create tool from
            name_override: Optional override for tool name
            description_override: Optional override for tool description

        Returns:
            Tool instance

        Raises:
            ValueError: If callable cannot be imported or is invalid
        """
        # If string provided, import the callable
        callable_obj = calling.import_callable(fn) if isinstance(fn, str) else fn
        module = inspect.getmodule(callable_obj)
        if not module:
            msg = f"Could not find module for callable: {callable_obj}"
            raise ImportError(msg)

        # Create dynamic subclass
        class DynamicTool(LLMCallableTool):
            # Store original callable for schema generation
            _original_callable = staticmethod(callable_obj)
            _import_path = f"{module.__name__}.{callable_obj.__qualname__}"  # type: ignore

            # Use provided name/description or derive from callable
            name = name_override or callable_obj.__name__
            description = (
                description_override
                or inspect.getdoc(callable_obj)
                or f"Tool from {callable_obj.__name__}"
            )

            async def execute(self, **params: Any) -> Any:
                """Execute the imported callable."""
                if calling.is_async_callable(callable_obj):
                    return await callable_obj(**params)
                return callable_obj(**params)

        return DynamicTool()


if __name__ == "__main__":

    def test(input_str: str, times: int = 2) -> str:
        """Multiply input string.

        Args:
            input_str: String to multiply
            times: Number of times to multiply (default: 2)
        """
        return input_str * times

    tool = LLMCallableTool.from_callable(test, name_override="Example Tool")
    print(tool.get_schema())
