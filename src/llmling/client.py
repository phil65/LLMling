"""High-level client interface for LLMling."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, TypeVar, cast, overload

from llmling.config.manager import ConfigManager
from llmling.context import default_registry as context_registry
from llmling.core import exceptions
from llmling.core.log import get_logger, setup_logging
from llmling.llm.registry import default_registry as llm_registry
from llmling.processors.registry import ProcessorRegistry
from llmling.task.concurrent import execute_concurrent
from llmling.task.executor import TaskExecutor
from llmling.task.manager import TaskManager


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    import os

    from llmling.task.models import TaskResult

logger = get_logger(__name__)


T = TypeVar("T")


class Registerable(Protocol):
    """Protocol for objects that can be registered."""

    def register(self, name: str, item: Any) -> None: ...


ComponentType = Literal["processor", "context", "provider"]


class LLMLingClient:
    """High-level client interface for interacting with LLMling."""

    def __init__(
        self,
        config_path: str | os.PathLike[str],
        *,
        log_level: int | None = None,
        validate_config: bool = True,
        components: dict[ComponentType, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            config_path: Path to YAML configuration file
            log_level: Optional logging level
            validate_config: Whether to validate configuration on load
            components: Optional components to register, organized by type
                Example:
                {
                    "processor": {"name": ProcessorConfig(...)},
                    "context": {"name": ContextLoader(...)},
                    "provider": {"name": LLMProvider(...)},
                }
        """
        if log_level is not None:
            setup_logging(level=log_level)

        self.config_path = config_path
        self.validate_config = validate_config
        self.components = components or {}

        # Initialize components as None
        self.config_manager: ConfigManager | None = None
        self.processor_registry: ProcessorRegistry | None = None
        self.executor: TaskExecutor | None = None
        self._manager: TaskManager | None = None
        self._initialized = False

    @property
    def manager(self) -> TaskManager:
        """Get the task manager, raising if not initialized."""
        if self._manager is None:
            msg = "Task manager not initialized"
            raise exceptions.LLMLingError(msg)
        return self._manager

    @classmethod
    def create(
        cls,
        config_path: str | os.PathLike[str],
        **kwargs: Any,
    ) -> LLMLingClient:
        """Create and initialize a client synchronously.

        Args:
            config_path: Path to configuration file
            **kwargs: Additional arguments for initialization

        Returns:
            Initialized client instance
        """
        client = cls(config_path, **kwargs)
        asyncio.run(client.startup())
        return client

    async def startup(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        try:
            # Initialize registries
            self.processor_registry = ProcessorRegistry()

            # Load configuration
            logger.info("Loading configuration from %s", self.config_path)
            from llmling.config.loading import load_config

            config = load_config(
                self.config_path,
                validate=self.validate_config,
            )
            self.config_manager = ConfigManager(config)

            # Register providers
            await self._register_providers()

            # Register components
            await self._register_components()

            # Start processor registry
            await self.processor_registry.startup()

            # Create executor and manager
            self.executor = TaskExecutor(
                context_registry=context_registry,
                processor_registry=self.processor_registry,
                provider_registry=llm_registry,
            )

            if self.config_manager is None:
                msg = "Configuration manager not initialized"
                raise exceptions.LLMLingError(msg)  # noqa: TRY301

            self._manager = TaskManager(self.config_manager.config, self.executor)
            self._initialized = True
            logger.info("Client initialized successfully")

        except Exception as exc:
            msg = "Failed to initialize client"
            raise exceptions.LLMLingError(msg) from exc

    async def _register_components(self) -> None:
        """Register all configured components."""
        registries: dict[ComponentType, Registerable] = {
            "processor": cast(Registerable, self.processor_registry),
            "context": cast(Registerable, context_registry),
            "provider": cast(Registerable, llm_registry),
        }

        for component_type, items in self.components.items():
            registry = registries.get(component_type)
            if registry is None:
                logger.warning("Unknown component type: %s", component_type)
                continue

            for name, item in items.items():
                try:
                    registry.register(name, item)
                    logger.debug(
                        "Registered %s: %s",
                        component_type,
                        name,
                    )
                except Exception:
                    logger.exception("Failed to register %s %s", component_type, name)

    async def shutdown(self) -> None:
        """Clean up resources."""
        if not self._initialized:
            return

        if self.processor_registry:
            await self.processor_registry.shutdown()
        self._initialized = False
        logger.info("Client shut down successfully")

    def _ensure_initialized(self) -> None:
        """Ensure client is initialized."""
        if not self._initialized:
            msg = "Client not initialized"
            raise exceptions.LLMLingError(msg)

    async def _register_providers(self) -> None:
        """Register all providers from configuration."""
        if not self.config_manager:
            msg = "Configuration not loaded"
            raise exceptions.LLMLingError(msg)

        # Register direct providers
        for provider_key in self.config_manager.config.llm_providers:
            llm_registry.register_provider(provider_key, "litellm")
            logger.debug("Registered provider: %s", provider_key)

        # Register provider groups
        for group_name, providers in self.config_manager.config.provider_groups.items():
            if providers:
                llm_registry.register_provider(group_name, "litellm")
                logger.debug("Registered provider group: %s", group_name)

    @overload
    async def execute(
        self,
        template: str,
        *,
        system_prompt: str | None = None,
        stream: Literal[True],
        **kwargs: Any,
    ) -> AsyncIterator[TaskResult]: ...

    @overload
    async def execute(
        self,
        template: str,
        *,
        system_prompt: str | None = None,
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> TaskResult: ...

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
            system_prompt: Optional system prompt
            stream: Whether to stream results
            **kwargs: Additional parameters for LLM

        Returns:
            Task result or async iterator of results if streaming
        """
        self._ensure_initialized()
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

    def execute_sync(
        self,
        template: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> TaskResult:
        """Execute a task template synchronously.

        Args:
            template: Name of template to execute
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for LLM

        Returns:
            Task result
        """
        return asyncio.run(
            self.execute(
                template,
                system_prompt=system_prompt,
                stream=False,
                **kwargs,
            )
        )

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
            templates: Template names to execute
            system_prompt: Optional system prompt
            max_concurrent: Maximum concurrent executions
            **kwargs: Additional parameters for LLM

        Returns:
            List of task results
        """
        self._ensure_initialized()
        return await execute_concurrent(
            self.manager,
            templates,
            system_prompt=system_prompt,
            max_concurrent=max_concurrent,
            **kwargs,
        )

    def execute_many_sync(
        self,
        templates: Sequence[str],
        **kwargs: Any,
    ) -> list[TaskResult]:
        """Execute multiple templates concurrently (synchronous version).

        Args:
            templates: Template names to execute
            **kwargs: Additional parameters passed to execute_many

        Returns:
            List of task results
        """
        return asyncio.run(self.execute_many(templates, **kwargs))

    async def stream(
        self,
        template: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[TaskResult]:
        """Stream results from a task template.

        This is a more explicit way to stream results compared to using
        execute() with stream=True.

        Args:
            template: Name of template to execute
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for LLM

        Returns:
            Async iterator of task results
        """
        self._ensure_initialized()
        try:
            async for result in self.manager.execute_template_stream(
                template,
                system_prompt=system_prompt,
                **kwargs,
            ):
                yield result
        except Exception as exc:
            msg = f"Failed to stream template {template}"
            raise exceptions.TaskError(msg) from exc

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()

    def __enter__(self) -> Self:
        """Synchronous context manager entry."""
        asyncio.run(self.startup())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Synchronous context manager exit."""
        asyncio.run(self.shutdown())


async def async_example() -> None:
    """Example of async usage."""
    async with LLMLingClient("src/llmling/resources/test.yml") as client:
        # Single execution
        result = await client.execute("quick_review")
        print("\nSingle execution result:")
        print("-" * 40)
        print(result.content)

        # Streaming execution
        print("\nStreaming execution:")
        print("-" * 40)
        async for chunk in client.stream("quick_review"):
            print(chunk.content, end="")

        # Concurrent execution
        print("\n\nConcurrent execution:")
        print("-" * 40)
        results = await client.execute_many(
            ["quick_review", "detailed_review"],
            max_concurrent=2,
        )
        for r in results:
            print(f"\nResult from {r.model}:")
            print(r.content)


def sync_example() -> None:
    """Example of synchronous usage."""
    with LLMLingClient.create("src/llmling/resources/test.yml") as client:
        # Single execution
        result = client.execute_sync("quick_review")
        print("\nSync execution result:")
        print("-" * 40)
        print(result.content)

        # Concurrent execution
        print("\nSync concurrent execution:")
        print("-" * 40)
        results = client.execute_many_sync(
            ["quick_review", "detailed_review"],
            max_concurrent=2,
        )
        for r in results:
            print(f"\nResult from {r.model}:")
            print(r.content)


if __name__ == "__main__":
    """Run both async and sync examples."""
    # Reset the registry before running examples
    llm_registry.reset()

    print("\nRunning async example...")
    print("=" * 50)
    asyncio.run(async_example())

    print("\nRunning sync example...")
    print("=" * 50)
    # Reset registry between examples
    llm_registry.reset()
    sync_example()
