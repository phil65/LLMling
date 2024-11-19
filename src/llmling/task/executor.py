"""Task execution system."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import logfire

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.llm.base import LLMConfig, Message
from llmling.task.models import TaskContext, TaskProvider, TaskResult
from llmling.tools.base import ToolRegistry


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling.context import ContextLoaderRegistry
    from llmling.llm.registry import ProviderRegistry
    from llmling.processors.registry import ProcessorRegistry


logger = get_logger(__name__)


class TaskExecutor:
    """Executes tasks using configured contexts and providers."""

    def __init__(
        self,
        context_registry: ContextLoaderRegistry,
        processor_registry: ProcessorRegistry,
        provider_registry: ProviderRegistry,
        tool_registry: ToolRegistry | None = None,
        *,
        default_timeout: int = 30,
        default_max_retries: int = 3,
    ) -> None:
        """Initialize the task executor.

        Args:
            context_registry: Registry of context loaders
            processor_registry: Registry of processors
            provider_registry: Registry of LLM providers
            tool_registry: Registry of LLM model tools
            default_timeout: Default timeout for LLM calls
            default_max_retries: Default retry count for LLM calls
        """
        self.context_registry = context_registry
        self.processor_registry = processor_registry
        self.provider_registry = provider_registry
        self.tool_registry = tool_registry or ToolRegistry()
        self.default_timeout = default_timeout
        self.default_max_retries = default_max_retries

    def _prepare_tool_config(
        self,
        task_context: TaskContext,
        task_provider: TaskProvider,
    ) -> dict[str, Any] | None:
        """Prepare tool configuration if tools are enabled."""
        if not self.tool_registry:
            return None

        available_tools = []

        # Add inherited tools from provider if enabled
        if (
            task_context.inherit_tools
            and task_provider.settings
            and task_provider.settings.tools
        ):
            available_tools.extend(task_provider.settings.tools)

        # Add task-specific tools
        if task_context.tools:
            available_tools.extend(
                self.tool_registry.get_schema(tool) for tool in task_context.tools
            )

        if not available_tools:
            return None

        return {
            "tools": available_tools,
            "tool_choice": (
                task_context.tool_choice
                or (
                    task_provider.settings.tool_choice if task_provider.settings else None
                )
                or "auto"
            ),
        }

    @logfire.instrument(
        "Executing task with provider {task_provider.name}, model {task_provider.model}"
    )
    async def execute(
        self,
        task_context: TaskContext,
        task_provider: TaskProvider,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> TaskResult:
        """Execute a task."""
        try:
            # Add tool configuration if available
            if tool_config := self._prepare_tool_config(task_context, task_provider):
                kwargs.update(tool_config)
            # Load and process context
            context_result = await self._load_context(task_context)

            # Prepare messages
            messages = self._prepare_messages(context_result.content, system_prompt)

            # Configure and create provider
            llm_config = self._create_llm_config(task_provider)
            provider = self.provider_registry.create_provider(
                task_provider.name,
                llm_config,
            )

            # Get completion with potential tool calls
            while True:
                completion = await provider.complete(messages, **kwargs)

                # Handle tool calls if present
                if completion.tool_calls:
                    tool_results = []
                    for tool_call in completion.tool_calls:
                        result = await self.tool_registry.execute(
                            tool_call.name,
                            **tool_call.parameters,
                        )
                        tool_results.append(result)

                    # Add tool results to messages
                    messages.append(
                        Message(
                            role="tool",
                            content=str(tool_results),
                            name="tool_results",
                        )
                    )
                    continue  # Get next completion

                # No tool calls, return final result
                return TaskResult(
                    content=completion.content,
                    model=completion.model,
                    context_metadata=context_result.metadata,
                    completion_metadata=completion.metadata,
                )

        except Exception as exc:
            msg = "Task execution failed"
            raise exceptions.TaskError(msg) from exc

    async def execute_stream(
        self,
        task_context: TaskContext,
        task_provider: TaskProvider,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[TaskResult]:
        """Execute a task with streaming results.

        Args:
            task_context: Context configuration
            task_provider: Provider configuration
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for LLM

        Yields:
            Streaming task results

        Raises:
            TaskError: If execution fails
        """
        try:
            # Load and process context
            context_result = await self._load_context(task_context)

            # Prepare messages
            messages = self._prepare_messages(
                context_result.content,
                system_prompt,
            )

            # Configure and create provider
            llm_config = self._create_llm_config(
                task_provider,
                streaming=True,
            )
            provider = self.provider_registry.create_provider(
                task_provider.name,
                llm_config,
            )

            # Stream completions
            async for completion in provider.complete_stream(messages, **kwargs):
                yield TaskResult(
                    content=completion.content,
                    model=completion.model,
                    context_metadata=context_result.metadata,
                    completion_metadata=completion.metadata,
                )

        except Exception as exc:
            msg = "Task streaming failed"
            raise exceptions.TaskError(msg) from exc

    async def _load_context(self, task_context: TaskContext) -> Any:
        """Load and process context.

        Args:
            task_context: Context configuration

        Returns:
            Processed context result

        Raises:
            TaskError: If context loading fails
        """
        try:
            # Get appropriate loader
            loader = self.context_registry.get_loader(task_context.context)

            # Load and process content
            return await loader.load(
                task_context.context,
                self.processor_registry,
            )

        except Exception as exc:
            msg = "Context loading failed"
            raise exceptions.TaskError(msg) from exc

    def _prepare_messages(
        self,
        content: str,
        system_prompt: str | None,
    ) -> list[Message]:
        """Prepare messages for LLM completion.

        Args:
            content: Context content
            system_prompt: Optional system prompt

        Returns:
            List of messages
        """
        messages: list[Message] = []

        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))

        messages.append(Message(role="user", content=content))
        return messages

    def _create_llm_config(
        self,
        provider: TaskProvider,
        *,
        streaming: bool = False,
    ) -> LLMConfig:
        """Create LLM configuration from provider settings."""
        provider_settings = (
            provider.settings.model_dump(exclude_none=True)
            if provider.settings is not None
            else {}
        )

        config_dict = {
            "model": provider.model,
            "provider_name": provider.name,  # Key for lookup
            "display_name": provider.display_name,  # Human readable name
            "timeout": self.default_timeout,
            "max_retries": self.default_max_retries,
            "streaming": streaming,
        }

        config_dict.update(provider_settings)
        return LLMConfig(**config_dict)
