"""Base class for implementing tools callable by an LLM via tool calling."""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from dataclasses import dataclass
import inspect
from typing import Any, ClassVar, Self

import schemez

from llmling.config.models import ToolHints  # noqa: TC001
from llmling.core.descriptors import classproperty


@dataclass
class LLMCallableTool[**P, TReturnType]:
    supported_mime_types: ClassVar[list[str]] = ["text/plain"]
    callable: Callable[P, TReturnType]
    name: str
    description: str = ""
    import_path: str | None = None
    schema_override: schemez.OpenAIFunctionDefinition | None = None
    hints: ToolHints | None = None

    @classmethod
    def from_callable(
        cls,
        fn: Callable[P, TReturnType] | str,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        schema_override: schemez.OpenAIFunctionDefinition | None = None,
        hints: ToolHints | None = None,
    ) -> Self:
        """Create a tool from a callable or import path."""
        if isinstance(fn, str):
            import_path = fn
            from llmling.utils import importing

            callable_obj = importing.import_callable(fn)
            name = getattr(callable_obj, "__name__", "unknown")
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
            schema_override=schema_override,
            hints=hints,
        )

    @classmethod
    def from_crewai_tool(
        cls,
        tool: Any,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        schema_override: schemez.OpenAIFunctionDefinition | None = None,
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
            schema_override=schema_override,
        )

    @classmethod
    def from_langchain_tool(
        cls,
        tool: Any,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        schema_override: schemez.OpenAIFunctionDefinition | None = None,
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
            schema_override=schema_override,
        )

    @classmethod
    def from_autogen_tool(
        cls,
        tool: Any,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        schema_override: schemez.OpenAIFunctionDefinition | None = None,
    ) -> Self:
        """Create a tool from a AutoGen tool."""
        try:
            from autogen_core import CancellationToken
            from autogen_core.tools import BaseTool
        except ImportError as e:
            msg = "autogent_core package not found."
            raise ImportError(msg) from e

        if not isinstance(tool, BaseTool):
            msg = f"Expected AutoGent BaseTool, got {type(tool)}"
            raise TypeError(msg)
        token = CancellationToken()

        input_model = tool.__class__.__orig_bases__[0].__args__[0]  # type: ignore

        name = name_override or tool.name or tool.__class__.__name__.removesuffix("Tool")
        description = (
            description_override
            or tool.description
            or inspect.getdoc(tool.__class__)
            or ""
        )

        async def wrapper(**kwargs: Any) -> Any:
            # Convert kwargs to the expected input model
            model = input_model(**kwargs)
            return await tool.run(model, cancellation_token=token)

        return cls.from_callable(
            wrapper,  # type: ignore
            name_override=name,
            description_override=description,
            schema_override=schema_override,
        )

    async def execute(self, *args: P.args, **kwargs: P.kwargs) -> TReturnType:
        """Execute the wrapped callable."""
        if inspect.iscoroutinefunction(self.callable):
            return await self.callable(*args, **kwargs)
        return self.callable(*args, **kwargs)

    def get_schema(self) -> schemez.OpenAIFunctionTool:
        """Get OpenAI function schema."""
        schema = schemez.create_schema(self.callable).model_dump_openai()
        schema["function"]["name"] = self.name
        schema["function"]["description"] = self.description
        if self.schema_override:
            schema["function"] = self.schema_override
        return schema


class BaseTool(LLMCallableTool):
    """Base class for complex tools requiring inheritance."""

    name: ClassVar[str]
    description: ClassVar[str]
    supported_mime_types: ClassVar[list[str]] = ["text/plain"]

    @classproperty  # type: ignore
    def import_path(cls) -> str:  # type: ignore  # noqa: N805
        """Get the import path of the tool class."""
        return f"{cls.__module__}.{cls.__qualname__}"  # type: ignore

    def get_schema(self) -> schemez.OpenAIFunctionTool:
        """Get OpenAI function schema."""
        schema = schemez.create_schema(self.execute).model_dump_openai()
        schema["function"]["name"] = self.name
        schema["function"]["description"] = self.description
        return schema

    async def execute(self, **params: Any) -> Any:
        """Execute the tool."""
        raise NotImplementedError


if __name__ == "__main__":
    import asyncio

    async def main():
        from crewai_tools import BraveSearchTool

        crew_ai_tool = BraveSearchTool()
        tool = LLMCallableTool[Any, Any].from_crewai_tool(crew_ai_tool)
        print(tool.get_schema())
        result = await tool.execute(query="What is the capital of France?")
        print(result)

    asyncio.run(main())
