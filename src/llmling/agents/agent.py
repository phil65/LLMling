"""LLMling integration with PydanticAI for AI-powered resource interaction."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
import inspect
from inspect import Parameter, Signature
from typing import TYPE_CHECKING, Any, cast

import logfire
from pydantic_ai import Agent as PydanticAgent, RunContext, messages
from pydantic_ai.result import RunResult, StreamedRunResult
from typing_extensions import TypeVar

from llmling.config.models import Config
from llmling.config.runtime import RuntimeConfig
from llmling.core.log import get_logger
from llmling.resources.models import LoadedResource  # noqa: TC001


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from py2openai import OpenAIFunctionTool
    from pydantic_ai.agent import models
    from pydantic_ai.dependencies import ToolParams, ToolPlainFunc

    from llmling.core.events import Event


logger = get_logger(__name__)

TResult = TypeVar("TResult", default=str)  # Type of results (str or Pydantic model)
T = TypeVar("T")  # For the return type


def _create_tool_wrapper(
    name: str,
    schema: OpenAIFunctionTool,
    original_callable: Callable[..., T | Awaitable[T]] | None = None,
) -> Callable[..., Awaitable[T]]:
    """Create a tool wrapper function with proper signature and type hints.

    Creates an async wrapper function that forwards calls to RuntimeConfig.execute_tool.
    If the original callable is provided, its signature and type hints are preserved.
    Otherwise, the signature is reconstructed from the OpenAI function schema.

    Args:
        name: Name of the tool to wrap
        schema: OpenAI function schema (from py2openai)
        original_callable: Optional original function to preserve signature from

    Returns:
        Async wrapper function with proper signature that delegates to execute_tool
    """
    # If we have the original callable, use its signature
    if original_callable:
        # Create parameters with original types
        sig = inspect.signature(original_callable)
        params = [
            Parameter(
                "ctx",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=RunContext[RuntimeConfig],
            ),
            *[
                Parameter(name, p.kind, annotation=p.annotation, default=p.default)
                for name, p in sig.parameters.items()
            ],
        ]
        return_annotation = sig.return_annotation
    else:
        # Fall back to schema-based parameters with Any types
        params = [
            Parameter(
                "ctx",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=RunContext[RuntimeConfig],
            )
        ]
        properties = schema["function"].get("parameters", {}).get("properties", {})
        for prop_name, info in properties.items():
            params.append(
                Parameter(
                    prop_name,
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=Any,
                    default=Parameter.empty if info.get("required") else None,
                )
            )
        return_annotation = Any

    # Create the signature
    sig = Signature(params, return_annotation=return_annotation)

    # Create the wrapper function
    async def tool_wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx = args[0]  # First arg is always context
        return await ctx.deps.execute_tool(name, **kwargs)

    # Apply the signature and metadata
    tool_wrapper.__signature__ = sig  # type: ignore
    tool_wrapper.__name__ = schema["function"]["name"]
    tool_wrapper.__doc__ = schema["function"]["description"]
    tool_wrapper.__annotations__ = {p.name: p.annotation for p in params}

    return tool_wrapper


class LLMlingAgent[TResult]:
    """Agent for AI-powered interaction with LLMling resources and tools.

    This agent integrates LLMling's resource system with PydanticAI's agent capabilities.
    It provides:
    - Access to resources through RuntimeConfig
    - Structured output support
    - Tool registration for resource operations
    - System prompt customization
    - Message history management

    Example:
        ```python
        # Simple text agent
        agent = LLMlingAgent(runtime)
        result = await agent.run("Load and summarize test.txt")
        print(result.data)  # Text summary

        # Agent with structured output
        class Analysis(BaseModel):
            summary: str
            complexity: int

        agent = LLMlingAgent[Analysis](
            runtime,
            result_type=Analysis,
        )
        result = await agent.run("Analyze test.txt")
        print(result.data.summary)  # Structured analysis
        ```
    """

    def __init__(
        self,
        runtime: RuntimeConfig,
        result_type: type[TResult] | None = None,
        *,
        model: models.Model | models.KnownModelName | None = None,
        system_prompt: str | Sequence[str] = (),
        name: str = "llmling-agent",
        retries: int = 1,
        result_tool_name: str = "final_result",
        result_tool_description: str | None = None,
        result_retries: int | None = None,
        defer_model_check: bool = False,
        **kwargs,
    ) -> None:
        """Initialize agent with runtime configuration.

        Args:
            runtime: Runtime configuration providing access to resources/tools
            result_type: Optional type for structured responses
            model: The default model to use (defaults to GPT-4)
            system_prompt: Static system prompts to use for this agent
            name: Name of the agent for logging
            retries: Default number of retries for failed operations
            result_tool_name: Name of the tool used for final result
            result_tool_description: Description of the final result tool
            result_retries: Max retries for result validation (defaults to retries)
            defer_model_check: Whether to defer model evaluation until first run
            kwargs: Additional arguments for PydanticAI agent
        """
        self._runtime = runtime

        # Initialize base PydanticAI agent
        self.pydantic_agent = PydanticAgent(
            model=model,
            result_type=result_type or str,  # Default to string responses
            system_prompt=system_prompt,
            deps_type=RuntimeConfig,  # Always use RuntimeConfig as deps
            retries=retries,
            result_tool_name=result_tool_name,
            result_tool_description=result_tool_description,
            result_retries=result_retries,
            defer_model_check=defer_model_check,
            **kwargs,
        )

        # Set up event handling
        self._runtime.add_event_handler(self)

        self._setup_default_tools()
        self._setup_runtime_tools()
        self._name = name
        logger.debug(
            "Initialized %s (model=%s, result_type=%s)",
            self._name,
            model,
            result_type or "str",
        )

    def _setup_runtime_tools(self) -> None:
        """Register all tools from runtime configuration."""
        for name, llm_tool in self.runtime.tools.items():
            schema = llm_tool.get_schema()
            wrapper: Callable[..., Any] = _create_tool_wrapper(name, schema)
            self.tool(wrapper)
            logger.debug(
                "Registered runtime tool: %s (signature: %s)",
                name,
                wrapper.__signature__,  # type: ignore
            )

    def _setup_default_tools(self) -> None:
        """Register default tools for resource operations."""

        @self.tool
        async def load_resource(
            ctx: RunContext[RuntimeConfig],
            uri: str,
        ) -> LoadedResource:
            """Load a resource by URI or name.

            Args:
                ctx: Context
                uri: Resource URI to load. Can be:
                    - Resource name: "test.txt"
                    - Full URI: "file:///test.txt"
                    - Local path: "/path/to/file.txt"
            """
            return await ctx.deps.load_resource_by_uri(uri)

        @self.tool
        async def list_resources(ctx: RunContext[RuntimeConfig]) -> Sequence[str]:
            """List available resources."""
            return ctx.deps.list_resources()

        @self.tool
        async def process_content(
            ctx: RunContext[RuntimeConfig],
            content: str,
            processor_name: str,
            **kwargs: Any,
        ) -> str:
            """Process content with a named processor.

            Args:
                ctx: Context
                content: Content to process
                processor_name: Name of processor to use
                **kwargs: Additional processor arguments
            """
            result = await ctx.deps.process_content(content, processor_name, **kwargs)
            return result.content

        @self.tool
        async def render_template(
            ctx: RunContext[RuntimeConfig],
            template: str,
            **variables: Any,
        ) -> str:
            """Render a template string using Jinja2.

            Args:
                ctx: Runtime context
                template: Template string to render
                **variables: Variables to use in template

            Returns:
                Rendered template string

            Example:
                "Hello {{ name }}!" with variables {"name": "World"}
                will return "Hello World!"
            """
            result = await ctx.deps.process_content(
                template,
                "jinja_template",
                **variables,
            )
            return result.content

    async def handle_event(self, event: Event) -> None:
        """Handle runtime events.

        Override this method to add custom event handling.
        """
        # Default implementation just logs
        logger.debug("Received event: %s", event)

    @logfire.instrument("Running agent")
    async def run(
        self,
        prompt: str,
        *,
        message_history: list[messages.Message] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> RunResult[TResult]:
        """Run agent with prompt and get response.

        Args:
            prompt: User query or instruction
            message_history: Optional previous messages for context
            model: Optional model override

        Returns:
            Result containing response and run information

        Raises:
            UnexpectedModelBehavior: If the model fails or behaves unexpectedly
        """
        try:
            result = await self.pydantic_agent.run(
                prompt,
                deps=self._runtime,
                message_history=message_history,
                model=model,
            )
            return cast(RunResult[TResult], result)
        except Exception:
            logger.exception("Agent run failed")
            raise

    @logfire.instrument("Streaming agent response")
    async def run_stream(
        self,
        prompt: str,
        *,
        message_history: list[messages.Message] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> StreamedRunResult[TResult, messages.Message]:
        """Run agent with prompt and stream response.

        Args:
            prompt: User query or instruction
            message_history: Optional previous messages for context
            model: Optional model override

        Returns:
            Streamed result

        Raises:
            UnexpectedModelBehavior: If the model fails or behaves unexpectedly
        """
        try:
            async with self.pydantic_agent.run_stream(
                prompt,
                deps=self._runtime,
                message_history=message_history,
                model=model,
            ) as result:
                return cast(StreamedRunResult[TResult, messages.Message], result)
        except Exception:
            logger.exception("Agent stream failed")
            raise

    def run_sync(
        self,
        prompt: str,
        *,
        message_history: list[messages.Message] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> RunResult[TResult]:
        """Run agent synchronously (convenience wrapper).

        Args:
            prompt: User query or instruction
            message_history:too Optional previous messages for context
            model: Optional model override

        Returns:
            Result containing response and run information

        Raises:
            UnexpectedModelBehavior: If the model fails or behaves unexpectedly
        """
        result = self.pydantic_agent.run_sync(
            prompt,
            deps=self._runtime,
            message_history=message_history,
            model=model,
        )
        return cast(RunResult[TResult], result)

    def tool(self, *args: Any, **kwargs: Any) -> Any:
        """Register a tool with the agent.

        Tools can access runtime through RunContext[RuntimeConfig].

        Example:
            ```python
            @agent.tool
            async def my_tool(ctx: RunContext[RuntimeConfig], arg: str) -> str:
                resource = await ctx.deps.load_resource(arg)
                return resource.content
            ```
        """
        return self.pydantic_agent.tool(*args, **kwargs)

    def tool_plain(self, func: ToolPlainFunc[ToolParams]) -> Any:
        """Register a plain tool with the agent.

        Plain tools don't receive runtime context.

        Example:
            ```python
            @agent.tool_plain
            def my_tool(arg: str) -> str:
                return arg.upper()
            ```
        """
        return self.pydantic_agent.tool_plain(func)

    def system_prompt(self, *args: Any, **kwargs: Any) -> Any:
        """Register a dynamic system prompt.

        System prompts can access runtime through RunContext[RuntimeConfig].

        Example:
            ```python
            @agent.system_prompt
            async def get_prompt(ctx: RunContext[RuntimeConfig]) -> str:
                resources = await ctx.deps.list_resources()
                return f"Available resources: {', '.join(resources)}"
            ```
        """
        return self.pydantic_agent.system_prompt(*args, **kwargs)

    def result_validator(self, *args: Any, **kwargs: Any) -> Any:
        """Register a result validator.

        Validators can access runtime through RunContext[RuntimeConfig].

        Example:
            ```python
            @agent.result_validator
            async def validate(ctx: RunContext[RuntimeConfig], result: str) -> str:
                if len(result) < 10:
                    raise ModelRetry("Response too short")
                return result
            ```
        """
        return self.pydantic_agent.result_validator(*args, **kwargs)

    @property
    def last_run_messages(self) -> list[messages.Message] | None:
        """Get messages from the last run."""
        return self.pydantic_agent.last_run_messages

    @property
    def runtime(self) -> RuntimeConfig:
        """Get the runtime configuration."""
        return self._runtime


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.DEBUG)

    async def main() -> None:
        cfg = Config.from_file("E:/mcp_zed.yml")
        runtime = RuntimeConfig.from_config(cfg)
        async with runtime as r:
            agent: LLMlingAgent[str] = LLMlingAgent(r, model="openai:gpt-3.5-turbo")
            result = await agent.run(
                "Get repo contents for user 'phil65' and repository 'llmling'"
            )
            print(result.data)

    asyncio.run(main())
