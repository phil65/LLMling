"""Helper functions for easy initialization and usage of LLMling."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Self, overload

from llmling.config.loading import load_config
from llmling.context import default_registry as context_registry
from llmling.core import exceptions
from llmling.core.log import get_logger, setup_logging
from llmling.llm.registry import default_registry as llm_registry
from llmling.processors.registry import ProcessorRegistry
from llmling.task.executor import TaskExecutor
from llmling.task.manager import TaskManager


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    import os

    from llmling.task.models import TaskResult


logger = get_logger(__name__)


class LLMLing:
    """Helper class for easy initialization and usage of LLMling."""

    def __init__(
        self,
        config_path: str | os.PathLike[str],
        *,
        log_level: int | None = None,
        validate_config: bool = True,
    ) -> None:
        """Initialize LLMling with configuration.

        Args:
            config_path: Path to YAML configuration file
            log_level: Optional logging level
            validate_config: Whether to validate configuration on load

        Raises:
            ConfigError: If configuration loading or validation fails
        """
        # Setup logging if level provided
        if log_level is not None:
            setup_logging(level=log_level)

        # Load and validate configuration
        logger.info("Loading configuration from %s", config_path)
        self.config = load_config(config_path, validate=validate_config)

        # Initialize registries and executor
        self.processor_registry = ProcessorRegistry()

        # Create executor with all registries
        self.executor = TaskExecutor(
            context_registry=context_registry,
            processor_registry=self.processor_registry,  # Use instance variable
            provider_registry=llm_registry,
        )

        # Create task manager
        self.manager = TaskManager(self.config, self.executor)

        # Register processors from config
        self._register_processors()

    def _register_processors(self) -> None:
        """Register processors from configuration."""
        for name, config in self.config.context_processors.items():
            self.processor_registry.register(name, config)
            logger.debug("Registered processor: %s", name)

    async def startup(self) -> None:
        """Initialize all components."""
        await self.processor_registry.startup()

    async def shutdown(self) -> None:
        """Clean up resources."""
        await self.processor_registry.shutdown()

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        await self.startup()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        await self.shutdown()

    @overload
    async def execute(
        self,
        template: str,
        *,
        system_prompt: str | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> TaskResult: ...

    @overload
    async def execute(
        self,
        template: str,
        *,
        system_prompt: str | None = None,
        stream: bool = True,
        **kwargs: Any,
    ) -> AsyncIterator[TaskResult]: ...

    async def execute(
        self,
        template: str,
        *,
        system_prompt: str | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> TaskResult | AsyncIterator[TaskResult]:
        """Execute a task template.

        Args:
            template: Name of template to execute
            system_prompt: Optional system prompt to use
            stream: Whether to stream results
            **kwargs: Additional parameters for LLM

        Returns:
            Task result or async iterator of results if streaming

        Raises:
            TaskError: If execution fails
        """
        try:
            if stream:
                return self.manager.execute_template_stream(
                    template,
                    system_prompt=system_prompt,
                    **kwargs,
                )
            return await self.manager.execute_template(
                template,
                system_prompt=system_prompt,
                **kwargs,
            )
        except Exception as exc:
            msg = f"Failed to execute template {template}"
            raise exceptions.TaskError(msg) from exc

    async def execute_many(
        self,
        templates: Sequence[str],
        *,
        system_prompt: str | None = None,
        max_concurrent: int = 3,
        **kwargs: Any,
    ) -> list[TaskResult]:
        """Execute multiple templates concurrently.

        Args:
            templates: Sequence of template names to execute
            system_prompt: Optional system prompt to use
            max_concurrent: Maximum number of concurrent executions
            **kwargs: Additional parameters for LLM

        Returns:
            List of task results

        Raises:
            TaskError: If execution fails
        """
        from llmling.task.concurrent import execute_concurrent

        try:
            return await execute_concurrent(
                self.manager,
                templates,
                system_prompt=system_prompt,
                max_concurrent=max_concurrent,
                **kwargs,
            )
        except Exception as exc:
            msg = "Concurrent execution failed"
            raise exceptions.TaskError(msg) from exc


# Example usage:
async def main() -> None:
    import logging

    # Using context manager (recommended)
    async with LLMLing("src/llmling/resources/test.yml", log_level=logging.INFO) as llm:
        # Single execution
        result = await llm.execute("quick_review", system_prompt="Be helpful")
        print(f"Result: {result.content}")

        # Streaming execution
        async for chunk in llm.execute("quick_review", stream=True):
            print(chunk.content, end="")

        # Concurrent execution
        results = await llm.execute_many(
            ["template1", "template2"],
            max_concurrent=2,
        )
        for r in results:
            print(f"Result from {r.model}: {r.content}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
