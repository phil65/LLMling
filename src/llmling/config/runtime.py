"""Runtime configuration handling.

This module provides the RuntimeConfig class which represents the fully initialized,
"live" state of a configuration, managing all runtime components and registries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import depkit
import logfire

from llmling.config.models import PathResource, Prompt, PromptConfig
from llmling.core import exceptions
from llmling.core.events import EventEmitter
from llmling.core.log import get_logger
from llmling.extensions.loaders import ToolsetLoader
from llmling.processors.registry import ProcessorRegistry
from llmling.prompts.function import create_prompt_from_callable
from llmling.prompts.registry import PromptRegistry
from llmling.resources import ResourceLoaderRegistry
from llmling.resources.loaders.path import PathResourceLoader
from llmling.resources.registry import ResourceRegistry
from llmling.tools.base import LLMCallableTool
from llmling.tools.registry import ToolRegistry


if TYPE_CHECKING:
    from collections.abc import Sequence
    import types

    from llmling.config.models import Config, Resource
    from llmling.core.events import RegistryEvents
    from llmling.prompts.models import PromptMessage
    from llmling.resources.models import LoadedResource


logger = get_logger(__name__)


class RuntimeConfig(EventEmitter):
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
        super().__init__()
        self._config = config
        self._loader_registry = loader_registry
        self._processor_registry = processor_registry
        self._resource_registry = resource_registry
        self._prompt_registry = prompt_registry
        self._tool_registry = tool_registry
        self._initialized = False

        settings = self._config.global_settings
        self._dep_manager = depkit.DependencyManager(
            prefer_uv=settings.prefer_uv,
            requirements=settings.requirements,
            extra_paths=settings.extra_paths,
            pip_index_url=settings.pip_index_url,
            scripts=settings.scripts,
        )

    def __enter__(self) -> Self:
        """Synchronous context manager entry."""
        self._dep_manager.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Synchronous context manager exit."""
        self._dep_manager.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> Self:
        """Initialize dependencies and registries."""
        await self._dep_manager.__aenter__()
        if not self._initialized:
            await self._initialize_registries()
            self._initialized = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Clean up dependencies and registries."""
        try:
            await self.shutdown()
        finally:
            await self._dep_manager.__aexit__(exc_type, exc_val, exc_tb)

    def _register_default_components(self) -> None:
        """Register all default components and config items."""
        from llmling.resources import (
            CallableResourceLoader,
            CLIResourceLoader,
            ImageResourceLoader,
            PathResourceLoader,
            SourceResourceLoader,
            TextResourceLoader,
        )

        self._loader_registry["path"] = PathResourceLoader
        self._loader_registry["text"] = TextResourceLoader
        self._loader_registry["cli"] = CLIResourceLoader
        self._loader_registry["source"] = SourceResourceLoader
        self._loader_registry["callable"] = CallableResourceLoader
        self._loader_registry["image"] = ImageResourceLoader

        for name, proc_config in self._config.context_processors.items():
            self._processor_registry[name] = proc_config

        for name, resource in self._config.resources.items():
            self._resource_registry[name] = resource

        for name, tool_config in self._config.tools.items():
            tool = LLMCallableTool.from_callable(
                tool_config.import_path,
                name_override=tool_config.name,
                description_override=tool_config.description,
            )
            self._tool_registry[name] = tool

        if self._config.toolsets:
            loader = ToolsetLoader()
            for name, tool in loader.load_items(self._config.toolsets).items():
                if name not in self._tool_registry:
                    self._tool_registry[name] = tool
                else:
                    logger.warning(
                        "Tool %s from toolset overlaps with configured tool",
                        name,
                    )

        for name, prompt_config in self._config.prompts.items():
            match prompt_config:
                case Prompt():
                    self._prompt_registry[name] = prompt_config
                case PromptConfig():
                    prompt = create_prompt_from_callable(
                        prompt_config.import_path,
                        name_override=prompt_config.name or name,
                        description_override=prompt_config.description,
                        template_override=prompt_config.template,
                    )
                    self._prompt_registry[name] = prompt

    async def _initialize_registries(self) -> None:
        """Initialize all registries."""
        self._register_default_components()
        await self.startup()

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
        async with runtime as initialized:
            return initialized

    @classmethod
    @logfire.instrument("Creating runtime configuration")
    def from_config(cls, config: Config) -> Self:
        """Create a fully initialized runtime config from static config.

        Args:
            config: Static configuration to initialize from

        Returns:
            Initialized runtime configuration
        """
        loader_registry = ResourceLoaderRegistry()
        processor_registry = ProcessorRegistry()
        resource_registry = ResourceRegistry(
            loader_registry=loader_registry,
            processor_registry=processor_registry,
        )
        prompt_registry = PromptRegistry()
        tool_registry = ToolRegistry()

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
        """Load a resource by name.

        Args:
            name: Name of the resource to load

        Returns:
            Loaded resource content and metadata

        Raises:
            ResourceError: If resource cannot be loaded
        """
        return await self._resource_registry.load(name)

    async def resolve_resource_uri(self, uri_or_name: str) -> tuple[str, Resource]:
        """Resolve a resource identifier to a proper URI and resource.

        Args:
            uri_or_name: Can be:
                - Resource name: "test.txt"
                - Full URI: "file:///test.txt"
                - Local path: "/path/to/file.txt"

        Returns:
            Tuple of (resolved URI, resource object)

        Raises:
            ResourceError: If resolution fails
        """
        logger.debug("Resolving resource identifier: %s", uri_or_name)

        # 1. If it's already a URI, use directly
        if "://" in uri_or_name:
            logger.debug("Using direct URI")
            loader = self._loader_registry.find_loader_for_uri(uri_or_name)
            name = loader.get_name_from_uri(uri_or_name)
            if name in self._resource_registry:
                return uri_or_name, self._resource_registry[name]
            # Create temporary resource for the URI
            resource: Resource = PathResource(path=uri_or_name)
            return uri_or_name, resource

        # 2. Try as resource name
        try:
            logger.debug("Trying as resource name")
            resource = self._resource_registry[uri_or_name]
            loader = self._loader_registry.get_loader(resource)
            loader = loader.create(resource, uri_or_name)  # Create instance
            uri = loader.create_uri(name=uri_or_name)
        except KeyError:
            pass
        else:
            return uri, resource

        # 3. If it looks like a path, try as file
        if "/" in uri_or_name or "\\" in uri_or_name or "." in uri_or_name:
            try:
                logger.debug("Trying as file path")
                resource = PathResource(path=uri_or_name)
                loader = PathResourceLoader.create(resource, uri_or_name)
                uri = loader.create_uri(name=uri_or_name)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to create file URI: %s", exc)
            else:
                return uri, resource
        msg = (
            f"Could not resolve resource {uri_or_name!r}. Expected resource name or path."
        )
        raise exceptions.ResourceError(msg)

    async def load_resource_by_uri(self, uri: str) -> LoadedResource:
        """Load a resource by URI."""
        try:
            resolved_uri, resource = await self.resolve_resource_uri(uri)
            loader = self._loader_registry.get_loader(resource)
            loader = loader.create(resource, loader.get_name_from_uri(resolved_uri))
            async for res in loader.load(processor_registry=self._processor_registry):
                return res  # Return first resource
            msg = "No resources loaded"
            raise exceptions.ResourceError(msg)  # noqa: TRY301
        except Exception as exc:
            msg = f"Failed to load resource from URI {uri}"
            raise exceptions.ResourceError(msg) from exc

    def list_resources(self) -> Sequence[str]:
        """List all available resource names.

        Returns:
            List of registered resource names
        """
        return self._resource_registry.list_items()

    def get_resource_uri(self, name: str) -> str:
        """Get URI for a resource.

        Args:
            name: Name of the resource

        Returns:
            URI for the resource

        Raises:
            ResourceError: If resource not found
        """
        return self._resource_registry.get_uri(name)

    def register_resource(
        self,
        name: str,
        resource: Resource,
        *,
        replace: bool = False,
    ) -> None:
        """Register a new resource.

        Args:
            name: Name for the resource
            resource: Resource to register
            replace: Whether to replace existing resource

        Raises:
            ResourceError: If name exists and replace=False
        """
        self._resource_registry.register(name, resource, replace=replace)

    def get_resource_loader(self, resource: Resource) -> Any:  # type: ignore[return]
        """Get loader for a resource type.

        Args:
            resource: Resource to get loader for

        Returns:
            Resource loader instance

        Raises:
            LoaderError: If no loader found for resource type
        """
        return self._loader_registry.get_loader(resource)

    # Tool Management
    def list_tools(self) -> Sequence[str]:
        """List all available tool names.

        Returns:
            List of registered tool names
        """
        return self._tool_registry.list_items()

    @property
    def tools(self) -> dict[str, LLMCallableTool]:
        """Get all registered tools.

        Returns:
            Dictionary mapping tool names to tools
        """
        return dict(self._tool_registry)

    async def execute_tool(self, name: str, **params: Any) -> Any:
        """Execute a tool by name.

        Args:
            name: Name of the tool to execute
            **params: Parameters to pass to the tool

        Returns:
            Tool execution result

        Raises:
            ToolError: If tool execution fails
        """
        return await self._tool_registry.execute(name, **params)

    def get_tool(self, name: str) -> LLMCallableTool:
        """Get a tool by name.

        Args:
            name: Name of the tool

        Returns:
            The tool

        Raises:
            ToolError: If tool not found
        """
        return self._tool_registry[name]

    def get_tools(self) -> Sequence[LLMCallableTool]:
        """Get all registered tools.

        Returns:
            List of all tools
        """
        return list(self._tool_registry.values())

    # Prompt Management
    def list_prompts(self) -> Sequence[str]:
        """List all available prompt names.

        Returns:
            List of registered prompt names
        """
        return self._prompt_registry.list_items()

    async def render_prompt(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Sequence[PromptMessage]:
        """Format a prompt with arguments.

        Args:
            name: Name of the prompt
            arguments: Optional arguments for formatting

        Returns:
            List of formatted messages

        Raises:
            LLMLingError: If prompt not found or formatting fails
        """
        try:
            prompt = self._prompt_registry[name]
            return await prompt.format(arguments)
        except KeyError as exc:
            msg = f"Prompt not found: {name}"
            raise exceptions.LLMLingError(msg) from exc
        except Exception as exc:
            msg = f"Failed to format prompt {name}: {exc}"
            raise exceptions.LLMLingError(msg) from exc

    def get_prompt(self, name: str) -> Prompt:
        """Get a prompt by name.

        Args:
            name: Name of the prompt

        Returns:
            The prompt

        Raises:
            LLMLingError: If prompt not found
        """
        try:
            return self._prompt_registry[name]
        except KeyError as exc:
            msg = f"Prompt not found: {name}"
            raise exceptions.LLMLingError(msg) from exc

    def get_prompts(self) -> Sequence[Prompt]:
        """Get all registered prompts.

        Returns:
            List of all prompts
        """
        return list(self._prompt_registry.values())

    # Registry Observation
    def add_resource_observer(self, observer: RegistryEvents[str, Resource]) -> None:
        """Add observer for resource changes.

        Args:
            observer: Observer to add
        """
        self._resource_registry.add_observer(observer)

    def add_prompt_observer(self, observer: RegistryEvents[str, Prompt]) -> None:
        """Add observer for prompt changes.

        Args:
            observer: Observer to add
        """
        self._prompt_registry.add_observer(observer)

    def add_tool_observer(self, observer: RegistryEvents[str, LLMCallableTool]) -> None:
        """Add observer for tool changes.

        Args:
            observer: Observer to add
        """
        self._tool_registry.add_observer(observer)

    def remove_resource_observer(self, observer: RegistryEvents[str, Resource]) -> None:
        """Remove resource observer.

        Args:
            observer: Observer to remove
        """
        self._resource_registry.remove_observer(observer)

    def remove_prompt_observer(self, observer: RegistryEvents[str, Prompt]) -> None:
        """Remove prompt observer.

        Args:
            observer: Observer to remove
        """
        self._prompt_registry.remove_observer(observer)

    def remove_tool_observer(
        self, observer: RegistryEvents[str, LLMCallableTool]
    ) -> None:
        """Remove tool observer.

        Args:
            observer: Observer to remove
        """
        self._tool_registry.remove_observer(observer)

    @property
    def original_config(self) -> Config:
        """Get the original static configuration.

        Returns:
            Original configuration
        """
        return self._config

    async def get_prompt_completions(
        self,
        current_value: str,
        argument_name: str,
        prompt_name: str,
        **options: Any,
    ) -> list[str]:
        """Get completions for a prompt argument.

        Args:
            current_value: Current input value
            argument_name: Name of the argument
            prompt_name: Name of the prompt
            **options: Additional options

        Returns:
            List of completion suggestions
        """
        return await self._prompt_registry.get_completions(
            current_value=current_value,
            argument_name=argument_name,
            prompt_name=prompt_name,
            **options,
        )

    async def get_resource_completions(
        self,
        uri: str,
        current_value: str,
        argument_name: str | None = None,
        **options: Any,
    ) -> list[str]:
        """Get completions for a resource.

        Args:
            uri: Resource URI
            current_value: Current input value
            argument_name: Optional argument name
            **options: Additional options

        Returns:
            List of completion suggestions
        """
        loader = self._loader_registry.find_loader_for_uri(uri)
        return await loader.get_completions(
            current_value=current_value,
            argument_name=argument_name,
            **options,
        )


if __name__ == "__main__":
    from llmling import Config

    cfg = Config.from_file("E:/mcp_zed.yml")  # type: ignore
    runtime = RuntimeConfig.from_config(cfg)
    with runtime as ctx:
        pass
