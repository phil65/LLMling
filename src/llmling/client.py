"""High-level client interface for LLMling."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, TypeVar, cast, overload

from llmling.config.loading import load_config
from llmling.config.manager import ConfigManager
from llmling.context import default_registry as context_registry
from llmling.core import exceptions
from llmling.core.log import get_logger, setup_logging
from llmling.llm.registry import default_registry as llm_registry
from llmling.processors.registry import ProcessorRegistry
from llmling.task.concurrent import execute_concurrent
from llmling.task.executor import TaskExecutor
from llmling.task.manager import TaskManager
from llmling.tools.base import ToolRegistry


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    import os

    from llmling.task.models import TaskResult

logger = get_logger(__name__)

T = TypeVar("T")


class Registerable(Protocol):
    """Protocol for objects that can be registered."""

    def register(self, name: str, item: Any) -> None: ...


ComponentType = Literal["processor", "context", "provider", "tool"]


class LLMLingClient:
    """High-level client interface for interacting with LLMling."""

    def __init__(
        self,
        config_path: str | os.PathLike[str],
        *,
        log_level: int | str | None = None,
        validate_config: bool = True,
        components: dict[ComponentType, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            config_path: Path to YAML configuration file
            log_level: Optional logging level
            validate_config: Whether to validate configuration on load
            components: Optional dictionary of components to register
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
        self.tool_registry = ToolRegistry()
        # Components will be initialized in startup
        self.config_manager: ConfigManager | None = None
        self._processor_registry: ProcessorRegistry | None = None
        self._executor: TaskExecutor | None = None
        self._manager: TaskManager | None = None
        self._initialized = False

    @property
    def manager(self) -> TaskManager:
        """Get the task manager, raising if not initialized."""
        self._ensure_initialized()
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

        Raises:
            LLMLingError: If initialization fails
        """
        client = cls(config_path, **kwargs)
        try:
            # Create new event loop for sync operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(client.startup())
                return client
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except Exception as exc:
            logger.exception("Failed to create client")
            msg = f"Failed to create client: {exc}"
            raise exceptions.LLMLingError(msg) from exc

    async def startup(self) -> None:
        """Initialize all components.

        Raises:
            LLMLingError: If initialization fails
        """
        if self._initialized:
            return

        try:
            # Initialize registries
            self._processor_registry = ProcessorRegistry()
            llm_registry.reset()  # Ensure clean state

            # Load configuration
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
            await self._processor_registry.startup()

            # Create executor and manager
            self._executor = TaskExecutor(
                context_registry=context_registry,
                processor_registry=self._processor_registry,
                provider_registry=llm_registry,
                tool_registry=self.tool_registry,
            )
            self._manager = TaskManager(self.config_manager.config, self._executor)

            self._initialized = True
            logger.debug("Client initialized successfully")

        except Exception as exc:
            logger.exception("Client initialization failed")
            await self.shutdown()  # Ensure cleanup on failure
            msg = f"Failed to initialize client: {exc}"
            raise exceptions.LLMLingError(msg) from exc

    async def _register_components(self) -> None:
        """Register all configured components.

        Raises:
            LLMLingError: If component registration fails
        """
        registries: dict[ComponentType, Registerable] = {
            "processor": cast(Registerable, self._processor_registry),
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
                    logger.debug("Registered %s: %s", component_type, name)
                except Exception as exc:
                    msg = f"Failed to register {component_type} {name}: {exc}"
                    raise exceptions.LLMLingError(msg) from exc

    async def _register_providers(self) -> None:
        """Register all providers from configuration.

        Raises:
            LLMLingError: If provider registration fails
        """
        if not self.config_manager:
            msg = "Configuration not loaded"
            raise exceptions.LLMLingError(msg)

        try:
            # Register direct providers
            for provider_key in self.config_manager.config.llm_providers:
                llm_registry.register_provider(provider_key, "litellm")
                logger.debug("Registered provider: %s", provider_key)

            # Register provider groups
            for (
                group_name,
                providers,
            ) in self.config_manager.config.provider_groups.items():
                if providers:
                    llm_registry.register_provider(group_name, "litellm")
                    logger.debug("Registered provider group: %s", group_name)
        except Exception as exc:
            msg = "Failed to register providers"
            raise exceptions.LLMLingError(msg) from exc

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

        Raises:
            TaskError: If execution fails
        """
        try:
            # Create new event loop for sync operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.execute(
                        template,
                        system_prompt=system_prompt,
                        stream=False,
                        **kwargs,
                    )
                )
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except Exception as exc:
            msg = f"Synchronous execution failed: {exc}"
            raise exceptions.TaskError(msg) from exc

    def execute_many_sync(
        self,
        templates: Sequence[str],
        **kwargs: Any,
    ) -> list[TaskResult]:
        """Execute multiple templates synchronously.

        Args:
            templates: Template names to execute
            **kwargs: Additional parameters passed to execute_many

        Returns:
            List of task results

        Raises:
            TaskError: If execution fails
        """
        try:
            # Create new event loop for sync operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.execute_many(templates, **kwargs))
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except Exception as exc:
            msg = f"Synchronous concurrent execution failed: {exc}"
            raise exceptions.TaskError(msg) from exc

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

        Raises:
            TaskError: If execution fails
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
            logger.exception("Task execution failed")
            msg = f"Failed to execute template {template}: {exc}"
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
            templates: Template names to execute
            system_prompt: Optional system prompt
            max_concurrent: Maximum concurrent executions
            **kwargs: Additional parameters for LLM

        Returns:
            List of task results

        Raises:
            TaskError: If execution fails
        """
        self._ensure_initialized()
        try:
            return await execute_concurrent(
                self.manager,
                templates,
                system_prompt=system_prompt,
                max_concurrent=max_concurrent,
                **kwargs,
            )
        except Exception as exc:
            logger.exception("Concurrent execution failed")
            msg = f"Concurrent execution failed: {exc}"
            raise exceptions.TaskError(msg) from exc

    async def stream(
        self,
        template: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[TaskResult]:
        """Stream results from a task template.

        Args:
            template: Name of template to execute
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for LLM

        Yields:
            Task results

        Raises:
            TaskError: If streaming fails
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
            logger.exception("Task streaming failed")
            msg = f"Failed to stream template {template}: {exc}"
            raise exceptions.TaskError(msg) from exc

    def _ensure_initialized(self) -> None:
        """Ensure client is initialized."""
        if not self._initialized:
            msg = "Client not initialized"
            raise exceptions.LLMLingError(msg)

    async def shutdown(self) -> None:
        """Clean up resources."""
        if not self._initialized:
            return

        try:
            if self._processor_registry:
                await self._processor_registry.shutdown()
        except Exception as exc:
            logger.exception("Error during shutdown")
            msg = f"Failed to shutdown client: {exc}"
            raise exceptions.LLMLingError(msg) from exc
        finally:
            self._initialized = False
            logger.info("Client shut down successfully")

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()

    def __enter__(self) -> Self:
        """Synchronous context manager entry."""
        try:
            # Create new event loop for sync operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.startup())
                return self
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except Exception as exc:
            msg = "Failed to enter context"
            raise exceptions.LLMLingError(msg) from exc

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Synchronous context manager exit."""
        try:
            # Create new event loop for sync operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.shutdown())
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except Exception as exc:
            logger.exception("Error during context exit")
            msg = "Failed to exit context"
            raise exceptions.LLMLingError(msg) from exc


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
