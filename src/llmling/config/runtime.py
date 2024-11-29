"""Runtime configuration handling.

This module provides the RuntimeConfig class which represents the fully initialized,
"live" state of a configuration, managing all runtime components and registries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import logfire

from llmling.config.models import Prompt, PromptConfig
from llmling.core.log import get_logger
from llmling.extensions.loaders import ToolsetLoader
from llmling.processors.registry import ProcessorRegistry
from llmling.prompts.function import create_prompt_from_callable
from llmling.prompts.registry import PromptRegistry
from llmling.resources import ResourceLoaderRegistry
from llmling.resources.registry import ResourceRegistry
from llmling.tools.base import LLMCallableTool
from llmling.tools.registry import ToolRegistry


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling.config.models import Config, Resource
    from llmling.core.events import RegistryEvents
    from llmling.prompts.models import PromptResult
    from llmling.resources.models import LoadedResource


logger = get_logger(__name__)


class RuntimeConfig:
    """Fully initialized runtime configuration.

    This represents the "live" state of a Config, with all components
    initialized and ready to use. It provides a clean interface to
    access and manage runtime resources without exposing internal registries.
    """

    def __init__(
        self,
        config: Config,
        *,
        loader_registry: ResourceLoaderRegistry,
        processor_registry: ProcessorRegistry,
        resource_registry: ResourceRegistry,
        prompt_registry: PromptRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """Initialize with config and registries.

        Args:
            config: Original static configuration
            loader_registry: Registry for resource loaders
            processor_registry: Registry for content processors
            resource_registry: Registry for resources
            prompt_registry: Registry for prompts
            tool_registry: Registry for tools
        """
        self._config = config
        self._loader_registry = loader_registry
        self._processor_registry = processor_registry
        self._resource_registry = resource_registry
        self._prompt_registry = prompt_registry
        self._tool_registry = tool_registry

    async def __aenter__(self) -> Self:
        """Initialize all components on context entry."""
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensure cleanup on context exit."""
        await self.shutdown()

    @classmethod
    async def create(cls, config: Config) -> Self:
        """Create and initialize a runtime configuration.

        This is a convenience method that ensures proper initialization
        when not using the async context manager.

        Args:
            config: Static configuration to initialize from

        Returns:
            Initialized runtime configuration
        """
        runtime = cls.from_config(config)
        await runtime.startup()
        return runtime

    @classmethod
    @logfire.instrument("Creating runtime configuration")
    def from_config(cls, config: Config) -> Self:
        """Create a fully initialized runtime config from static config.

        Args:
            config: Static configuration to initialize from

        Returns:
            Initialized runtime configuration
        """
        # Create core registries first
        loader_registry = ResourceLoaderRegistry()
        processor_registry = ProcessorRegistry()

        # Create dependent registries
        resource_registry = ResourceRegistry(
            loader_registry=loader_registry,
            processor_registry=processor_registry,
        )
        prompt_registry = PromptRegistry(
            loader_registry=loader_registry,
            processor_registry=processor_registry,
        )
        tool_registry = ToolRegistry()

        # Register default loaders
        from llmling.resources import (
            CallableResourceLoader,
            CLIResourceLoader,
            ImageResourceLoader,
            PathResourceLoader,
            SourceResourceLoader,
            TextResourceLoader,
        )

        loader_registry["text"] = TextResourceLoader
        loader_registry["path"] = PathResourceLoader
        loader_registry["cli"] = CLIResourceLoader
        loader_registry["source"] = SourceResourceLoader
        loader_registry["callable"] = CallableResourceLoader
        loader_registry["image"] = ImageResourceLoader

        # Initialize from config
        for name, proc_config in config.context_processors.items():
            processor_registry[name] = proc_config

        for name, resource in config.resources.items():
            resource_registry[name] = resource

        # Register explicit tools
        for name, tool_config in config.tools.items():
            tool = LLMCallableTool.from_callable(
                tool_config.import_path,
                name_override=tool_config.name,
                description_override=tool_config.description,
            )
            tool_registry[name] = tool

        # Load tools from toolsets
        if config.toolsets:
            loader = ToolsetLoader()
            for name, tool in loader.load_items(config.toolsets).items():
                if name not in tool_registry:
                    tool_registry[name] = tool
                else:
                    logger.warning(
                        "Tool %s from toolset overlaps with configured tool",
                        name,
                    )

        for name, prompt_config in config.prompts.items():
            match prompt_config:
                case Prompt():
                    prompt_registry[name] = prompt_config
                case PromptConfig():
                    # Create prompt from function
                    prompt = create_prompt_from_callable(
                        prompt_config.import_path,
                        name_override=prompt_config.name or name,
                        description_override=prompt_config.description,
                        template_override=prompt_config.template,
                    )
                    prompt_registry[name] = prompt

        return cls(
            config=config,
            loader_registry=loader_registry,
            processor_registry=processor_registry,
            resource_registry=resource_registry,
            prompt_registry=prompt_registry,
            tool_registry=tool_registry,
        )

    async def startup(self) -> None:
        """Start all runtime components."""
        await self._processor_registry.startup()
        await self._tool_registry.startup()
        await self._resource_registry.startup()
        await self._prompt_registry.startup()

    async def shutdown(self) -> None:
        """Shut down all runtime components."""
        await self._prompt_registry.shutdown()
        await self._resource_registry.shutdown()
        await self._tool_registry.shutdown()
        await self._processor_registry.shutdown()

    # Resource Management
    async def load_resource(self, name: str) -> LoadedResource:
        """Load a resource by name."""
        return await self._resource_registry.load(name)

    async def load_resource_by_uri(self, uri: str) -> LoadedResource:
        """Load a resource by URI."""
        return await self._resource_registry.load_by_uri(uri)

    def list_resources(self) -> Sequence[str]:
        """List all available resource names."""
        return self._resource_registry.list_items()

    def get_resource_uri(self, name: str) -> str:
        """Get URI for a resource."""
        return self._resource_registry.get_uri(name)

    def register_resource(
        self,
        name: str,
        resource: Resource,
        *,
        replace: bool = False,
    ) -> None:
        """Register a new resource."""
        self._resource_registry.register(name, resource, replace=replace)

    def get_resource_loader(self, resource: Resource) -> Any:  # type: ignore[return]
        """Get loader for a resource type."""
        return self._loader_registry.get_loader(resource)

    # Tool Management
    def list_tools(self) -> Sequence[str]:
        """List all available tool names."""
        return self._tool_registry.list_items()

    async def execute_tool(self, name: str, **params: Any) -> Any:
        """Execute a tool by name."""
        return await self._tool_registry.execute(name, **params)

    def get_tool(self, name: str) -> LLMCallableTool:
        """Get a tool by name."""
        return self._tool_registry[name]

    def get_tools(self) -> Sequence[LLMCallableTool]:
        """Get all registered tools."""
        return list(self._tool_registry.values())

    # Prompt Management
    def list_prompts(self) -> Sequence[str]:
        """List all available prompt names."""
        return self._prompt_registry.list_items()

    def get_prompt(self, name: str) -> Prompt:
        """Get a prompt by name."""
        return self._prompt_registry[name]

    def get_prompts(self) -> Sequence[Prompt]:
        """Get all registered prompts."""
        return list(self._prompt_registry.values())

    async def render_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> PromptResult:
        """Render a prompt with arguments."""
        return await self._prompt_registry.render(name, arguments)

    # Registry Observation
    def add_resource_observer(self, observer: RegistryEvents[str, Resource]) -> None:
        """Add observer for resource changes."""
        self._resource_registry.add_observer(observer)

    def add_prompt_observer(self, observer: RegistryEvents[str, Prompt]) -> None:
        """Add observer for prompt changes."""
        self._prompt_registry.add_observer(observer)

    def add_tool_observer(self, observer: RegistryEvents[str, LLMCallableTool]) -> None:
        """Add observer for tool changes."""
        self._tool_registry.add_observer(observer)

    def remove_resource_observer(self, observer: RegistryEvents[str, Resource]) -> None:
        """Remove resource observer."""
        self._resource_registry.remove_observer(observer)

    def remove_prompt_observer(self, observer: RegistryEvents[str, Prompt]) -> None:
        """Remove prompt observer."""
        self._prompt_registry.remove_observer(observer)

    def remove_tool_observer(
        self, observer: RegistryEvents[str, LLMCallableTool]
    ) -> None:
        """Remove tool observer."""
        self._tool_registry.remove_observer(observer)

    @property
    def original_config(self) -> Config:
        """Get the original static configuration."""
        return self._config
