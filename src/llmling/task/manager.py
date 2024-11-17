"""Task template management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.task.models import TaskContext, TaskProvider, TaskResult


if TYPE_CHECKING:
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
        """Initialize task manager.

        Args:
            config: Application configuration
            executor: Task executor
        """
        self.config = config
        self.executor = executor

    async def execute_template(
        self,
        template_name: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> TaskResult:
        """Execute a task template.

        Args:
            template_name: Name of template to execute
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for LLM

        Returns:
            Task execution result

        Raises:
            TaskError: If execution fails
        """
        template = self._get_template(template_name)
        context = self._resolve_context(template)
        provider = self._resolve_provider(template)

        task_context = TaskContext(
            context=context,
            processors=context.processors,
            inherit_tools=template.inherit_tools,
        )

        task_provider = TaskProvider(
            name=provider.name,
            model=provider.model,
            settings=template.settings,
        )

        return await self.executor.execute(
            task_context,
            task_provider,
            system_prompt=system_prompt,
            **kwargs,
        )

    async def execute_template_stream(
        self,
        template_name: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[TaskResult]:
        """Execute a task template with streaming results.

        Args:
            template_name: Name of template to execute
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for LLM

        Yields:
            Streaming task results

        Raises:
            TaskError: If execution fails
        """
        template = self._get_template(template_name)
        context = self._resolve_context(template)
        provider = self._resolve_provider(template)

        task_context = TaskContext(
            context=context,
            processors=context.processors,
            inherit_tools=template.inherit_tools,
        )

        task_provider = TaskProvider(
            name=provider.name,
            model=provider.model,
            settings=template.settings,
        )

        async for result in self.executor.execute_stream(
            task_context,
            task_provider,
            system_prompt=system_prompt,
            **kwargs,
        ):
            yield result

    def _get_template(self, name: str) -> TaskTemplate:
        """Get a task template by name.

        Args:
            name: Template name

        Returns:
            Task template

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
            Resolved context

        Raises:
            TaskError: If context resolution fails
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

            msg = f"Context {template.context} not found in contexts or context groups"
            raise exceptions.TaskError(msg)

        except exceptions.TaskError:
            raise
        except Exception as exc:
            msg = f"Failed to resolve context {template.context}"
            raise exceptions.TaskError(msg) from exc

    def _resolve_provider(self, template: TaskTemplate) -> LLMProviderConfig:
        """Resolve provider from template.

        Args:
            template: Task template

        Returns:
            Resolved provider configuration

        Raises:
            TaskError: If provider resolution fails
        """
        try:
            # Check direct provider first
            if template.provider in self.config.llm_providers:
                return self.config.llm_providers[template.provider]

            # Check provider groups
            if template.provider in self.config.provider_groups:
                # For now, just take the first provider in the group
                provider_name = self.config.provider_groups[template.provider][0]
                return self.config.llm_providers[provider_name]

            msg = (
                f"Provider {template.provider} not found in providers or provider groups"
            )
            raise exceptions.TaskError(msg)

        except exceptions.TaskError:
            raise
        except Exception as exc:
            msg = f"Failed to resolve provider {template.provider}"
            raise exceptions.TaskError(msg) from exc

        except Exception as exc:
            msg = "Provider resolution failed"
            raise exceptions.TaskError(msg) from exc
