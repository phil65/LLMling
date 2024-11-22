"""High-level client interface for LLMling."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar, overload

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
from llmling.tools.registry import ToolRegistry


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    import os
    from types import TracebackType

    from llmling.core.baseregistry import BaseRegistry
    from llmling.task.models import TaskResult

logger = get_logger(__name__)

T = TypeVar("T")

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
        """
        if log_level is not None:
            setup_logging(level=log_level)

        self.config_path = config_path
        self.validate_config = validate_config
        self.components = components or {}
        self.tool_registry = ToolRegistry()
        # Components will be initialized in startup
        self.config_manager: ConfigManager | None = None
        self._processor_registry = ProcessorRegistry()
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
        """Create and initialize a client synchronously."""
        client = cls(config_path, **kwargs)
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(client.startup())
                return client
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except Exception as exc:
            msg = f"Failed to create client: {exc}"
            raise exceptions.LLMLingError(msg) from exc

    async def startup(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        try:
            # Initialize registries
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

            # Create executor with empty tool registry
            self._executor = TaskExecutor(
                context_registry=context_registry,
                processor_registry=self._processor_registry,
                provider_registry=llm_registry,
                tool_registry=self.tool_registry,
                config_manager=self.config_manager,
            )

            # Create manager (will handle tool registration)
            self._manager = TaskManager(self.config_manager.config, self._executor)

            self._initialized = True
            logger.debug("Client initialized successfully")

        except Exception as exc:
            logger.exception("Client initialization failed")
            await self.shutdown()
            msg = f"Failed to initialize client: {exc}"
            raise exceptions.LLMLingError(msg) from exc

    def execute_sync(
        self,
        template: str,
        *,  # Force keyword arguments
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        **llm_params: Any,
    ) -> TaskResult:
        """Execute a task template synchronously.

        Args:
            template: Name of template to execute
            context: Task-specific context variables
            system_prompt: Optional system prompt
            **llm_params: Additional parameters for LLM
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.execute(
                        template,
                        context=context,
                        system_prompt=system_prompt,
                        llm_params=llm_params,
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
        *,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        max_concurrent: int = 3,
        **llm_params: Any,
    ) -> list[TaskResult]:
        """Execute multiple templates synchronously."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.execute_many(
                        templates,
                        context=context,
                        system_prompt=system_prompt,
                        max_concurrent=max_concurrent,
                        llm_params=llm_params,
                    )
                )
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
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        stream: Literal[True],
        llm_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[TaskResult]: ...

    @overload
    async def execute(
        self,
        template: str,
        *,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        stream: Literal[False] = False,
        llm_params: dict[str, Any] | None = None,
    ) -> TaskResult: ...

    async def execute(
        self,
        template: str,
        *,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        stream: bool = False,
        llm_params: dict[str, Any] | None = None,
    ) -> TaskResult | AsyncIterator[TaskResult]:
        """Execute a task template.

        Args:
            template: Name of template to execute
            context: Task-specific context variables
            system_prompt: Optional system prompt
            stream: Whether to stream results
            llm_params: Parameters for the LLM provider
        """
        self._ensure_initialized()
        try:
            if stream:
                return self.manager.execute_template_stream(
                    template,
                    context=context,
                    system_prompt=system_prompt,
                    llm_params=llm_params or {},
                )
            return await self.manager.execute_template(
                template,
                context=context,
                system_prompt=system_prompt,
                llm_params=llm_params or {},
            )
        except Exception as exc:
            msg = f"Failed to execute template {template}"
            raise exceptions.TaskError(msg) from exc

    async def execute_many(
        self,
        templates: Sequence[str],
        *,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        max_concurrent: int = 3,
        llm_params: dict[str, Any] | None = None,
    ) -> list[TaskResult]:
        """Execute multiple templates concurrently."""
        self._ensure_initialized()
        try:
            return await execute_concurrent(
                self.manager,
                templates,
                context=context,
                system_prompt=system_prompt,
                max_concurrent=max_concurrent,
                llm_params=llm_params or {},
            )
        except Exception as exc:
            msg = "Concurrent execution failed"
            raise exceptions.TaskError(msg) from exc

    async def stream(
        self,
        template: str,
        *,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        llm_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[TaskResult]:
        """Stream results from a task template."""
        self._ensure_initialized()
        try:
            async for result in self.manager.execute_template_stream(
                template,
                context=context,
                system_prompt=system_prompt,
                llm_params=llm_params or {},
            ):
                yield result
        except Exception as exc:
            msg = f"Failed to stream template {template}"
            raise exceptions.TaskError(msg) from exc

    async def _register_components(self) -> None:
        """Register all configured components."""
        registries: dict[ComponentType, BaseRegistry[str, Any]] = {
            "processor": self._processor_registry,
            "context": context_registry,
            "provider": llm_registry,
            "tool": self.tool_registry,
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
        """Register all providers from configuration."""
        if not self.config_manager:
            msg = "Configuration not loaded"
            raise exceptions.LLMLingError(msg)

        try:
            # Register direct providers
            for provider_key in self.config_manager.config.llm_providers:
                llm_registry[provider_key] = "litellm"
                logger.debug("Registered provider: %s", provider_key)

            # Register provider groups
            for (
                group_name,
                providers,
            ) in self.config_manager.config.provider_groups.items():
                if providers:
                    llm_registry[group_name] = "litellm"
                    logger.debug("Registered provider group: %s", group_name)
        except Exception as exc:
            msg = "Failed to register providers"
            raise exceptions.LLMLingError(msg) from exc

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

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        await self.shutdown()

    def __enter__(self) -> Self:
        """Synchronous context manager entry."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.startup())
                return self
            finally:
                asyncio.set_event_loop(None)
        except Exception as exc:
            msg = "Failed to enter context"
            raise exceptions.LLMLingError(msg) from exc

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Synchronous context manager exit."""
        try:
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
