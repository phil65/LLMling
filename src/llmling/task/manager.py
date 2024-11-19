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

        # Register tools only if they exist in config
        if self.config.tools:
            self._register_tools()

    def _register_tools(self) -> None:
        """Register tools from configuration."""
        if not self.config.tools:
            return

        for tool_id, tool_config in self.config.tools.items():
            self.executor.tool_registry.register_path(
                import_path=tool_config.import_path,
                name=tool_config.name or tool_id,
                description=tool_config.description,
            )

    def _prepare_task(
        self,
        template_name: str,
        system_prompt: str | None = None,
    ) -> tuple[TaskContext, TaskProvider]:
        """Prepare task context and provider."""
        template = self._get_template(template_name)
        context = self._resolve_context(template)
        provider_name, provider_config = self._resolve_provider(template)

        # Create task context with proper tool configuration
        task_context = TaskContext(
            context=context,
            processors=context.processors,
            inherit_tools=template.inherit_tools or False,
            tools=template.tools,
            tool_choice=template.tool_choice,
        )

        # Create task provider with settings
        task_provider = TaskProvider(
            name=provider_name,
            model=provider_config.model,
            display_name=provider_config.name,
            settings=template.settings,
        )

        return task_context, task_provider

    async def execute_template(
        self,
        template_name: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> TaskResult:
        """Execute a task template."""
        try:
            task_context, task_provider = self._prepare_task(template_name, system_prompt)
            return await self.executor.execute(
                task_context,
                task_provider,
                system_prompt=system_prompt,
                **kwargs,
            )
        except Exception as exc:
            logger.exception("Task execution failed")
            msg = f"Task execution failed for template {template_name}: {exc}"
            raise exceptions.TaskError(msg) from exc

    async def execute_template_stream(
        self,
        template_name: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[TaskResult]:
        """Execute a task template with streaming results."""
        try:
            task_context, task_provider = self._prepare_task(template_name, system_prompt)
            async for result in self.executor.execute_stream(
                task_context,
                task_provider,
                system_prompt=system_prompt,
                **kwargs,
            ):
                yield result
        except Exception as exc:
            logger.exception("Task streaming failed")
            msg = f"Task streaming failed: {exc}"
            raise exceptions.TaskError(msg) from exc

    def _get_template(self, name: str) -> TaskTemplate:
        """Get a task template by name."""
        try:
            return self.config.task_templates[name]
        except KeyError as exc:
            msg = f"Task template not found: {name}"
            raise exceptions.TaskError(msg) from exc

    def _resolve_context(self, template: TaskTemplate) -> Context:
        """Resolve context from template."""
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
        msg = f"Context {template.context} not found in contexts or context groups"
        raise exceptions.TaskError(msg)

    def _resolve_provider(self, template: TaskTemplate) -> tuple[str, LLMProviderConfig]:
        """Resolve provider from template.

        Returns:
            Tuple of (provider_name, provider_config)
        """
        try:
            # Check direct provider first
            if template.provider in self.config.llm_providers:
                return template.provider, self.config.llm_providers[template.provider]

            # Check provider groups
            if template.provider in self.config.provider_groups:
                provider_name = self.config.provider_groups[template.provider][0]
                return provider_name, self.config.llm_providers[provider_name]
        except exceptions.TaskError:
            raise
        except Exception as exc:
            msg = f"Failed to resolve provider {template.provider}"
            raise exceptions.TaskError(msg) from exc
        msg = f"Provider {template.provider} not found in providers or provider groups"
        raise exceptions.TaskError(msg)
