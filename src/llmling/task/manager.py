"""Task template management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.task.models import TaskContext, TaskProvider, TaskResult


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling.config import Config, Context, LLMProviderConfig, TaskTemplate
    from llmling.task.executor import TaskExecutor

logger = get_logger(__name__)


class TaskManager:
    """Manages task templates and execution."""

    def __init__(
        self,
        config: Config,
        executor: TaskExecutor,
    ) -> None:
        """Initialize task manager."""
        self.config = config
        self.executor = executor
        if self.config.tools:
            self._register_tools()

    def _register_tools(self) -> None:
        """Register tools from configuration."""
        if not self.config.tools:
            logger.debug("No tools defined in configuration")
            return

        for tool_id, tool_config in self.config.tools.items():
            logger.debug("Registering tool: %s with config: %s", tool_id, tool_config)
            self.executor.tool_registry[tool_id] = {
                "import_path": tool_config.import_path,
                "name": tool_config.name or tool_id,
                "description": tool_config.description,
            }
            logger.debug("Successfully registered tool: %s", tool_id)

    async def execute_template(
        self,
        template_name: str,
        *,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        llm_params: dict[str, Any] | None = None,
    ) -> TaskResult:
        """Execute a task template.

        Args:
            template_name: Template to execute
            context: Task-specific context variables
            system_prompt: Optional system prompt
            llm_params: Parameters for the LLM provider

        Returns:
            Task execution result

        Raises:
            TaskError: If execution fails
        """
        try:
            llm_params = llm_params or {}
            task_context, task_provider = self._prepare_task(
                template_name,
                runtime_context=context,
            )
            return await self.executor.execute(
                task_context,
                task_provider,
                system_prompt=system_prompt,
                llm_params=llm_params,
            )
        except Exception as exc:
            msg = f"Task execution failed for template {template_name}"
            raise exceptions.TaskError(msg) from exc

    async def execute_template_stream(
        self,
        template_name: str,
        *,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        llm_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[TaskResult]:
        """Execute a task template with streaming results.

        Args:
            template_name: Template to execute
            context: Task-specific context variables
            system_prompt: Optional system prompt
            llm_params: Parameters for the LLM provider

        Yields:
            Streaming task results

        Raises:
            TaskError: If execution fails
        """
        try:
            llm_params = llm_params or {}
            task_context, task_provider = self._prepare_task(
                template_name,
                runtime_context=context,
            )
            async for result in self.executor.execute_stream(
                task_context,
                task_provider,
                system_prompt=system_prompt,
                llm_params=llm_params,
            ):
                yield result
        except Exception as exc:
            msg = f"Task streaming failed for template {template_name}"
            raise exceptions.TaskError(msg) from exc

    def _prepare_task(
        self,
        template_name: str,
        *,
        runtime_context: dict[str, Any] | None = None,
    ) -> tuple[TaskContext, TaskProvider]:
        """Prepare task context and provider.

        Args:
            template_name: Template name
            runtime_context: Runtime context variables

        Returns:
            Tuple of (TaskContext, TaskProvider)

        Raises:
            TaskError: If template not found or configuration invalid
        """
        template = self._get_template(template_name)
        base_context = self._resolve_context(template)
        provider_name, provider_config = self._resolve_provider(template)

        task_context = TaskContext(
            context=base_context,
            processors=base_context.processors,
            inherit_tools=template.inherit_tools or False,
            tools=template.tools,
            runtime_context=runtime_context,
            tool_choice=template.tool_choice,
        )

        task_provider = TaskProvider(
            name=provider_name,
            model=provider_config.model,
            display_name=provider_config.name,
            settings=template.settings,
        )

        return task_context, task_provider

    def _get_template(self, name: str) -> TaskTemplate:
        """Get a task template by name.

        Args:
            name: Template name

        Returns:
            Task template configuration

        Raises:
            TaskError: If template not found
        """
        try:
            return self.config.task_templates[name]
        except KeyError as exc:
            msg = f"Task template not found: {name}"
            raise exceptions.TaskError(msg) from exc

    def _resolve_context(self, template: TaskTemplate) -> Context:
        """Resolve context from template.

        Args:
            template: Task template

        Returns:
            Context configuration

        Raises:
            TaskError: If context not found
        """
        try:
            # Check direct context first
            if template.context in self.config.contexts:
                return self.config.contexts[template.context]

            # Check context groups
            if template.context in self.config.context_groups:
                # For now, just take the first context in the group
                context_name = self.config.context_groups[template.context][0]
                return self.config.contexts[context_name]
        except exceptions.TaskError:
            raise
        except Exception as exc:
            msg = f"Failed to resolve context {template.context}"
            raise exceptions.TaskError(msg) from exc
        else:
            msg = f"Context {template.context} not found in contexts or context groups"
            raise exceptions.TaskError(msg)

    def _resolve_provider(self, template: TaskTemplate) -> tuple[str, LLMProviderConfig]:
        """Resolve provider from template.

        Args:
            template: Task template

        Returns:
            Tuple of (provider_name, provider_config)

        Raises:
            TaskError: If provider not found
        """
        try:
            # Check direct provider first
            if template.provider in self.config.llm_providers:
                return template.provider, self.config.llm_providers[template.provider]

            # Check provider groups
            if template.provider in self.config.provider_groups:
                # Take first provider in group
                provider_name = self.config.provider_groups[template.provider][0]
                return provider_name, self.config.llm_providers[provider_name]
        except exceptions.TaskError:
            raise
        except Exception as exc:
            msg = f"Failed to resolve provider {template.provider}"
            raise exceptions.TaskError(msg) from exc
        else:
            msg = (
                f"Provider {template.provider} not found in providers or provider groups"
            )
            raise exceptions.TaskError(msg)
