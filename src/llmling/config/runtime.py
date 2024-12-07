"""Runtime configuration handling.

This module provides the RuntimeConfig class which represents the fully initialized,
"live" state of a configuration, managing all runtime components and registries.
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any, Literal, Self

import depkit
import logfire
from upath import UPath

from llmling.config.manager import ConfigManager
from llmling.config.models import (
    PathResource,
)
from llmling.config.utils import prepare_runtime, toolset_config_to_toolset
from llmling.core import exceptions
from llmling.core.events import EventEmitter
from llmling.core.log import get_logger
from llmling.core.typedefs import ProcessingStep
from llmling.processors.jinjaprocessor import Jinja2Processor
from llmling.processors.registry import ProcessorRegistry
from llmling.prompts.models import DynamicPrompt, FilePrompt, PromptMessage, StaticPrompt
from llmling.prompts.registry import PromptRegistry
from llmling.prompts.utils import extract_function_info
from llmling.resources import ResourceLoaderRegistry
from llmling.resources.loaders.path import PathResourceLoader
from llmling.resources.registry import ResourceRegistry
from llmling.tools.base import LLMCallableTool
from llmling.tools.registry import ToolRegistry
from llmling.utils import importing


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Sequence
    import os
    import types

    from llmling.completions.types import CompletionFunction
    from llmling.config.models import Config, Resource
    from llmling.core.events import RegistryEvents
    from llmling.processors.base import ProcessorResult
    from llmling.prompts.models import BasePrompt
    from llmling.resources.models import LoadedResource


logger = get_logger(__name__)
RegistryType = Literal["resource", "prompt", "tool"]


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
        # Register builtin processors
        proc = Jinja2Processor(config.global_settings.jinja_environment)
        self._processor_registry.register("jinja_template", proc)
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
        # Initialize registries if not already done
        import asyncio

        if not self._initialized:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self._initialize_registries())
                self._initialized = True
            finally:
                loop.close()
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
            self._tool_registry[name] = LLMCallableTool.from_callable(
                tool_config.import_path,
                name_override=tool_config.name,
                description_override=tool_config.description,
            )

        self._initialize_toolsets()

        for name, prompt_config in self._config.prompts.items():
            if isinstance(prompt_config, DynamicPrompt):
                # Convert completion function import paths to actual functions
                if completions := prompt_config.completions:
                    completion_funcs: dict[str, CompletionFunction] = {}
                    for arg_name, import_path in completions.items():
                        try:
                            func = importing.import_callable(import_path)
                            completion_funcs[arg_name] = func
                        except ValueError:
                            msg = "Failed to import completion function for %s: %s"
                            logger.warning(msg, arg_name, import_path)
                    prompt_config.completions = completion_funcs  # type: ignore
                args, desc = extract_function_info(
                    prompt_config.import_path, prompt_config.completions
                )
                prompt_config.arguments = args
                if not prompt_config.description:
                    prompt_config.description = desc

            self._prompt_registry[name] = prompt_config

    def _initialize_toolsets(self) -> None:
        """Initialize toolsets from config."""
        for name, config in self._config.toolsets.items():
            try:
                toolset = toolset_config_to_toolset(config)
                # Get tool prefix
                prefix = f"{config.namespace}." if config.namespace else f"{name}."

                # Register all tools
                for tool in toolset.get_llm_callable_tools():
                    tool_name = f"{prefix}{tool.name}"
                    if tool_name in self._tool_registry:
                        msg = "Tool %s from toolset %s overlaps with existing tool"
                        logger.warning(msg, tool.name, name)
                        continue
                    self._tool_registry[tool_name] = tool

            except Exception:
                logger.exception("Failed to load toolset: %s", name)

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
    @asynccontextmanager
    async def open(
        cls,
        source: str | os.PathLike[str] | Config,
        *,
        validate: bool = True,
        strict: bool = False,
    ) -> AsyncIterator[RuntimeConfig]:
        """Create and manage a runtime configuration asynchronously.

        This is the primary way to create and use a RuntimeConfig. It ensures proper
        initialization and cleanup of all resources and provides an async context
        manager interface.

        Args:
            source: Either a path to a configuration file or a Config object.
                   File paths can be strings or PathLike objects.
            validate: Whether to validate the configuration. When True, performs
                     additional checks beyond basic schema validation.
            strict: Whether to raise exceptions on validation warnings. Only
                   applicable when validate=True.

        Yields:
            Fully initialized RuntimeConfig instance

        Raises:
            ConfigError: If configuration is invalid or validation fails in strict mode
            TypeError: If source is neither a path nor a Config object
            OSError: If configuration file cannot be accessed

        Example:
            ```python
            # Using a config file:
            async with RuntimeConfig.open("config.yml") as runtime:
                resource = await runtime.load_resource("example")

            # Using an existing Config object:
            config = Config(...)
            async with RuntimeConfig.open(config) as runtime:
                resource = await runtime.load_resource("example")
            ```

        Note:
            The context manager ensures that all resources are properly initialized
            before use and cleaned up afterwards, even if an error occurs.
        """
        runtime = prepare_runtime(cls, source, validate=validate, strict=strict)
        async with runtime as r:
            yield r

    @classmethod
    @contextmanager
    def open_sync(
        cls,
        source: str | os.PathLike[str] | Config,
        *,
        validate: bool = True,
        strict: bool = False,
    ) -> Iterator[RuntimeConfig]:
        """Create and manage a runtime configuration synchronously.

        This is the synchronous version of open(). It provides the same functionality
        but uses a standard synchronous context manager interface. Use this if you
        don't need async functionality.

        Args:
            source: Either a path to a configuration file or a Config object.
                   File paths can be strings or PathLike objects.
            validate: Whether to validate the configuration. When True, performs
                     additional checks beyond basic schema validation.
            strict: Whether to raise exceptions on validation warnings. Only
                   applicable when validate=True.

        Yields:
            Fully initialized RuntimeConfig instance

        Raises:
            ConfigError: If configuration is invalid or validation fails in strict mode
            TypeError: If source is neither a path nor a Config object
            OSError: If configuration file cannot be accessed

        Example:
            ```python
            # Using a config file:
            with RuntimeConfig.open_sync("config.yml") as runtime:
                resource = runtime.load_resource_sync("example")

            # Using an existing Config object:
            config = Config(...)
            with RuntimeConfig.open_sync(config) as runtime:
                resource = runtime.load_resource_sync("example")
            ```

        Note:
            The context manager ensures that all resources are properly initialized
            before use and cleaned up afterwards, even if an error occurs.
        """
        runtime = prepare_runtime(cls, source, validate=validate, strict=strict)
        with runtime:
            yield runtime

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> Self:
        """Convenience function to directly create runtime config from a file.

        Args:
            path: Path to the config file

        Returns:
            Initialized runtime configuration
        """
        manager = ConfigManager.load(path)
        return cls.from_config(manager.config)

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
            name = loader.get_name_from_uri(resolved_uri)
            loader = loader.create(resource, name)
            async for res in loader.load(processor_registry=self._processor_registry):
                return res  # Return first resource
            msg = "No resources loaded"
            raise exceptions.ResourceError(msg)  # noqa: TRY301
        except Exception as exc:
            msg = f"Failed to load resource from URI {uri}"
            raise exceptions.ResourceError(msg) from exc

    def list_resource_names(self) -> Sequence[str]:
        """List all available resource names.

        Returns:
            List of registered resource names
        """
        return self._resource_registry.list_items()

    def list_resource_uris(self) -> Sequence[str]:
        """List all available resource URIs.

        Returns:
            List of registered resource names
        """
        return [res.uri for res in self._resource_registry.values() if res.uri]

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

    def get_resource(self, name: str) -> Resource:
        """Get a resource configuration by name.

        Args:
            name: Name of the resource to get

        Returns:
            The resource configuration

        Raises:
            ResourceError: If resource not found
        """
        return self._resource_registry[name]

    def get_resources(self) -> Sequence[Resource]:
        """Get all registered resources.

        Returns:
            List of all resources
        """
        return list(self._resource_registry.values())

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
    def list_tool_names(self) -> Sequence[str]:
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
    def list_prompt_names(self) -> Sequence[str]:
        """List all available prompt names.

        Returns:
            List of registered prompt names
        """
        return self._prompt_registry.list_items()

    def register_dynamic_prompt(
        self,
        name: str,
        fn: str | Callable[..., Any],
        *,
        description: str | None = None,
        template: str | None = None,
        replace: bool = False,
    ) -> None:
        """Register a function-based prompt.

        Args:
            name: Name to register under
            fn: Function or import path (e.g., "module.submodule.function")
            description: Optional description (extracted from docstring if not provided)
            template: Optional template for formatting output
            replace: Whether to replace existing prompt

        Example:
            >>> def get_info(url: str) -> str:
            ...     '''Fetch info from URL.'''
            ...     return "info"
            ...
            >>> runtime.register_dynamic_prompt("fetch", get_info)
            >>> # or
            >>> runtime.register_dynamic_prompt(
            ...     "github",
            ...     "my_module.get_github_info",
            ...     description="Fetch GitHub repository information"
            ... )
        """
        try:
            prompt = DynamicPrompt.from_callable(
                fn,
                name_override=name,
                description_override=description,
                template_override=template,
            )
            self._prompt_registry.register(name, prompt, replace=replace)

        except Exception as exc:
            msg = f"Failed to register dynamic prompt {name!r}: {exc}"
            raise exceptions.LLMLingError(msg) from exc

    def register_static_prompt(
        self,
        name: str,
        content: str,
        description: str | None = None,
        *,
        replace: bool = False,
    ) -> None:
        r"""Register a static prompt with text content.

        A static prompt is a simple text template that can be used as-is or with
        Python's string formatting (using {placeholder} syntax).

        Args:
            name: Name to register under
            content: Prompt text content
            description: Optional description of what the prompt does
            replace: Whether to replace existing prompt

        Example:
            >>> runtime.register_static_prompt(
            ...     "summarize",
            ...     "Please summarize the following text in 3 bullet points:\\n{text}",
            ...     description="Create a bullet-point summary",
            ... )
            >>> runtime.register_static_prompt(
            ...     "greet",
            ...     "Hello {name}! Welcome to {company}.",
            ...     description="Simple greeting template",
            ... )

        Raises:
            LLMLingError: If registration fails
        """
        try:
            # Create standard user message from content
            messages = [PromptMessage(role="user", content=content)]

            # Use content's first line as description if none provided
            if not description:
                first_line = content.split("\n", 1)[0]
                description = (
                    first_line[:100] + "..." if len(first_line) > 100 else first_line  # noqa: PLR2004
                )

            prompt = StaticPrompt(
                name=name,
                description=description,
                messages=messages,
            )
            self._prompt_registry.register(name, prompt, replace=replace)

        except Exception as exc:
            msg = f"Failed to register static prompt {name!r}: {exc}"
            raise exceptions.LLMLingError(msg) from exc

    def register_file_prompt(
        self,
        name: str,
        path: str | os.PathLike[str],
        *,
        description: str,
        output_format: Literal["text", "markdown", "jinja2"] = "text",
        watch: bool = False,
        replace: bool = False,
    ) -> None:
        """Register a file-based prompt.

        Args:
            name: Name to register under
            path: Path to the prompt file
            description: Human-readable description
            output_format: Format of the file content
            watch: Whether to watch file for changes
            replace: Whether to replace existing prompt

        Example:
            >>> runtime.register_file_prompt(
            ...     "analyze",
            ...     "prompts/analysis.md",
            ...     description="Code analysis prompt",
            ...     format="markdown"
            ... )
        """
        try:
            if not UPath(path).exists():
                msg = f"Prompt file not found: {path}"
                raise FileNotFoundError(msg)  # noqa: TRY301

            prompt = FilePrompt(
                name=name,
                path=path,
                description=description,
                format=output_format,
                watch=watch,
            )
            self._prompt_registry.register(name, prompt, replace=replace)

        except Exception as exc:
            msg = f"Failed to register file prompt {name!r}: {exc}"
            raise exceptions.LLMLingError(msg) from exc

    def register_prompt(
        self,
        name: str,
        prompt: BasePrompt,
        *,
        replace: bool = False,
    ) -> None:
        """Register a pre-configured prompt.

        Note:
            The specialized registration methods should be preferred:
            - register_dynamic_prompt() for function-based prompts
            - register_static_prompt() for message-based prompts
            - register_file_prompt() for file-based prompts

        This method is provided for advanced use cases where you need
        to register an already configured prompt instance.

        Args:
            name: Name to register under
            prompt: Configured prompt instance
            replace: Whether to replace existing prompt
        """
        if isinstance(prompt, dict):
            if "type" not in prompt:
                msg = "Missing prompt type in configuration"
                raise exceptions.ConfigError(msg)
            from llmling.prompts.models import DynamicPrompt, FilePrompt, StaticPrompt

            match prompt["type"]:
                case "text":
                    prompt_obj: BasePrompt = StaticPrompt.model_validate(prompt)
                case "function":
                    prompt_obj = DynamicPrompt.model_validate(prompt)
                case "file":
                    prompt_obj = FilePrompt.model_validate(prompt)
                case _:
                    msg = f"Unknown prompt type: {prompt['type']}"
                    raise exceptions.ConfigError(msg)

        self._prompt_registry.register(name, prompt_obj, replace=replace)

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

    def get_prompt(self, name: str) -> BasePrompt:
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

    def get_prompts(self) -> Sequence[BasePrompt]:
        """Get all registered prompts.

        Returns:
            List of all prompts
        """
        return list(self._prompt_registry.values())

    async def process_content(
        self,
        content: str,
        processor_name: str,
        **kwargs: Any,
    ) -> ProcessorResult:
        """Process content with a named processor.

        Args:
            content: Content to process
            processor_name: Name of processor to use
            **kwargs: Additional processor arguments

        Returns:
            Processing result

        Raises:
            ProcessorError: If processing fails
        """
        return await self._processor_registry.process(
            content, [ProcessingStep(name=processor_name, kwargs=kwargs)]
        )

    # Registry Observation

    def add_observer(
        self,
        observer: RegistryEvents[str, Any],
        registry_type: RegistryType,
    ) -> None:
        """Add an observer for registry changes.

        Args:
            observer: Observer to add
            registry_type: Type of registry to observe
        """
        match registry_type:
            case "resource":
                self._resource_registry.add_observer(observer)
            case "prompt":
                self._prompt_registry.add_observer(observer)
            case "tool":
                self._tool_registry.add_observer(observer)

    def remove_observer(
        self,
        observer: RegistryEvents[str, Any],
        registry_type: RegistryType,
    ) -> None:
        """Remove a registry observer.

        Args:
            observer: Observer to remove
            registry_type: Type of registry to remove from
        """
        match registry_type:
            case "resource":
                self._resource_registry.remove_observer(observer)
            case "prompt":
                self._prompt_registry.remove_observer(observer)
            case "tool":
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
    with RuntimeConfig.open_sync("E:/mcp_zed.yml"):
        pass
