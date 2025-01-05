"""Base class for implementing tools callable by an LLM via tool calling."""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from dataclasses import dataclass
import inspect
from typing import Any, ClassVar, Protocol, Self, runtime_checkable

import py2openai

from llmling.core.descriptors import classproperty


@runtime_checkable
class ToolProtocol(Protocol):
    """Protocol defining the interface that all tools must implement."""

    name: str
    description: str
    import_path: str
    supported_mime_types: ClassVar[list[str]]

    def get_schema(self) -> py2openai.OpenAIFunctionTool: ...

    @property
    def system_prompt(self) -> str: ...

    async def execute(self, **params: Any) -> Any: ...


@dataclass
class LLMCallableTool[TReturnType]:
    supported_mime_types: ClassVar[list[str]] = ["text/plain"]
    callable: Callable[..., TReturnType]
    name: str
    description: str = ""
    import_path: str | None = None

    @classmethod
    def from_callable(
        cls,
        fn: Callable[..., TReturnType] | str,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
    ) -> Self:
        """Create a tool from a callable or import path."""
        if isinstance(fn, str):
            import_path = fn
            from llmling.utils import importing

            callable_obj = importing.import_callable(fn)
            name = callable_obj.__name__
            import_path = fn
        else:
            callable_obj = fn
            module = fn.__module__
            if hasattr(fn, "__qualname__"):  # Regular function
                name = fn.__name__
                import_path = f"{module}.{fn.__qualname__}"
            else:  # Instance with __call__ method
                name = fn.__class__.__name__
                import_path = f"{module}.{fn.__class__.__qualname__}"

        return cls(
            callable=callable_obj,
            name=name_override or name,
            description=description_override or inspect.getdoc(callable_obj) or "",
            import_path=import_path,
        )

    @classmethod
    def from_crewai_tool(
        cls,
        tool: Any,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
    ) -> Self:
        """Allows importing crewai / langchain tools."""
        try:
            from crewai.tools import BaseTool as CrewAiBaseTool
        except ImportError as e:
            msg = "crewai package not found. Please install it with 'pip install crewai'"
            raise ImportError(msg) from e

        if not isinstance(tool, CrewAiBaseTool):
            msg = f"Expected CrewAI BaseTool, got {type(tool)}"
            raise TypeError(msg)

        return cls.from_callable(
            tool._run,
            name_override=name_override or tool.__class__.__name__.removesuffix("Tool"),
            description_override=description_override or tool.description,
        )

    @classmethod
    def from_langchain_tool(
        cls,
        tool: Any,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
    ) -> Self:
        """Create a tool from a LangChain tool."""
        try:
            from langchain_core.tools import BaseTool as LangChainBaseTool
        except ImportError as e:
            msg = "langchain-core package not found."
            raise ImportError(msg) from e

        if not isinstance(tool, LangChainBaseTool):
            msg = f"Expected LangChain BaseTool, got {type(tool)}"
            raise TypeError(msg)

        return cls.from_callable(
            tool.invoke,
            name_override=name_override or tool.name,
            description_override=description_override or tool.description,
        )

    async def execute(self, **params: Any) -> Any:
        """Execute the wrapped callable."""
        if inspect.iscoroutinefunction(self.callable):
            return await self.callable(**params)
        return self.callable(**params)

    def get_schema(self) -> py2openai.OpenAIFunctionTool:
        """Get OpenAI function schema."""
        schema = py2openai.create_schema(self.callable).model_dump_openai()
        schema["function"]["name"] = self.name
        schema["function"]["description"] = self.description
        return schema

    @property
    def system_prompt(self) -> str:
        """Tool-specific system prompt."""
        return ""


class BaseTool(LLMCallableTool):
    """Base class for complex tools requiring inheritance."""

    name: ClassVar[str]
    description: ClassVar[str]
    supported_mime_types: ClassVar[list[str]] = ["text/plain"]

    @classproperty  # type: ignore
    def import_path(cls) -> str:  # noqa: N805
        """Get the import path of the tool class."""
        return f"{cls.__module__}.{cls.__qualname__}"  # type: ignore

    def get_schema(self) -> py2openai.OpenAIFunctionTool:
        """Get OpenAI function schema."""
        schema = py2openai.create_schema(self.execute).model_dump_openai()
        schema["function"]["name"] = self.name
        schema["function"]["description"] = self.description
        return schema

    @property
    def system_prompt(self) -> str:
        """Tool-specific system prompt."""
        return ""

    async def execute(self, **params: Any) -> Any:
        """Execute the tool."""
        raise NotImplementedError


if __name__ == "__main__":
    import asyncio

    async def main():
        from crewai_tools import BraveSearchTool

        crew_ai_tool = BraveSearchTool()
        tool = LLMCallableTool[Any].from_crewai_tool(crew_ai_tool)
        print(tool.get_schema())
        result = await tool.execute(query="What is the capital of France?")
        print(result)

    asyncio.run(main())
