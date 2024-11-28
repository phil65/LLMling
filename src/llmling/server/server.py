"""MCP server implementation."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Self

import mcp
from mcp.server import Server
from mcp.types import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    CompleteResult,
    Completion,
    CompletionArgument,
    GetPromptResult,
    TextContent,
)
from pydantic import AnyUrl

from llmling import config_resources
from llmling.config.models import PathResource, SourceResource
from llmling.core.log import get_logger
from llmling.processors.registry import ProcessorRegistry
from llmling.prompts.models import ArgumentType, Prompt
from llmling.prompts.registry import PromptRegistry
from llmling.resources import ResourceLoaderRegistry
from llmling.resources.registry import ResourceRegistry
from llmling.server import conversions
from llmling.server.log import configure_server_logging, run_logging_processor
from llmling.server.observers import PromptObserver, ResourceObserver, ToolObserver
from llmling.tools.registry import ToolRegistry


if TYPE_CHECKING:
    from collections.abc import Coroutine
    import os

    from llmling.config.models import Config

logger = get_logger(__name__)

DEFAULT_NAME = "llmling-server"


class LLMLingServer:
    def __init__(
        self,
        config: Config,
        *,
        name: str = DEFAULT_NAME,
        loader_registry: ResourceLoaderRegistry | None = None,
        processor_registry: ProcessorRegistry | None = None,
        prompt_registry: PromptRegistry | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        """Initialize server."""
        self.config = config
        self.name = name

        # Initialize registries
        self.loader_registry = loader_registry or ResourceLoaderRegistry()
        self.processor_registry = processor_registry or ProcessorRegistry()
        self.prompt_registry = prompt_registry or PromptRegistry()
        self.tool_registry = tool_registry or ToolRegistry()

        # Create resource registry with dependencies
        self.resource_registry = ResourceRegistry(
            loader_registry=self.loader_registry,
            processor_registry=self.processor_registry,
        )

        # Register default resource loaders if using new registry
        if loader_registry is None:
            from llmling.resources import (
                CallableResourceLoader,
                CLIResourceLoader,
                ImageResourceLoader,
                PathResourceLoader,
                SourceResourceLoader,
                TextResourceLoader,
            )

            self.loader_registry["text"] = TextResourceLoader
            self.loader_registry["path"] = PathResourceLoader
            self.loader_registry["cli"] = CLIResourceLoader
            self.loader_registry["source"] = SourceResourceLoader
            self.loader_registry["callable"] = CallableResourceLoader
            self.loader_registry["image"] = ImageResourceLoader

        # Create MCP server
        self.server = Server(name)
        self._setup_handlers()
        self._setup_observers()
        logger.debug("Server initialized with name: %s", name)
        self._tasks: set[asyncio.Task[Any]] = set()

    def _create_task(self, coro: Coroutine[None, None, Any]) -> asyncio.Task[Any]:
        """Create and track an asyncio task."""
        task: asyncio.Task[Any] = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _setup_observers(self) -> None:
        """Set up registry observers."""
        # Create observers
        self.resource_observer = ResourceObserver(self)
        self.prompt_observer = PromptObserver(self)
        self.tool_observer = ToolObserver(self)

        # Register with registries
        self.resource_registry.add_observer(self.resource_observer.events)
        self.prompt_registry.add_observer(self.prompt_observer.events)
        self.tool_registry.add_observer(self.tool_observer.events)

    def _setup_handlers(self) -> None:
        """Register MCP protocol handlers."""

        @self.server.set_logging_level()
        async def handle_set_level(level: mcp.LoggingLevel) -> None:
            """Handle logging level changes."""
            # Map MCP levels to Python logging levels
            level_map = {
                "debug": logging.DEBUG,
                "info": logging.INFO,
                "notice": logging.INFO,
                "warning": logging.WARNING,
                "error": logging.ERROR,
                "critical": logging.CRITICAL,
                "alert": logging.CRITICAL,
                "emergency": logging.CRITICAL,
            }
            try:
                python_level = level_map[level]
                logger.setLevel(python_level)

                # Notify via log message
                await self.current_session.send_log_message(
                    level="info",
                    data=f"Log level set to {level}",
                    logger=self.name,
                )
            except Exception as exc:
                error = mcp.McpError("Error setting the log level")
                error.error = mcp.ErrorData(code=INTERNAL_ERROR, message=str(exc))
                raise error from exc

        @self.server.list_tools()
        async def handle_list_tools() -> list[mcp.types.Tool]:
            """Handle tools/list request."""
            return [conversions.to_mcp_tool(tool) for tool in self.tool_registry.values()]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict[str, Any] | None = None
        ) -> list[TextContent]:
            """Handle tools/call request."""
            try:
                result = await self.tool_registry.execute(name, **(arguments or {}))
                return [TextContent(type="text", text=str(result))]
            except Exception as exc:
                logger.exception("Tool execution failed: %s", name)
                error_msg = f"Tool execution failed: {exc}"
                return [TextContent(type="text", text=error_msg)]

        @self.server.list_prompts()
        async def handle_list_prompts() -> list[mcp.types.Prompt]:
            """Handle prompts/list request."""
            return [
                conversions.to_mcp_prompt(prompt)
                for prompt in self.prompt_registry.values()
            ]

        @self.server.get_prompt()
        async def handle_get_prompt(
            name: str, arguments: dict[str, str] | None = None
        ) -> GetPromptResult:
            """Handle prompts/get request."""
            result = await self.prompt_registry.render(name, arguments or {})
            messages = [conversions.to_mcp_message(msg) for msg in result.messages]
            return GetPromptResult(description=f"Prompt: {name}", messages=messages)

        @self.server.list_resources()
        async def handle_list_resources() -> list[mcp.types.Resource]:
            """Handle resources/list request."""
            resources = []

            # Use registry directly since we already registered resources in startup
            for name in self.resource_registry:
                try:
                    loaded = await self.resource_registry.load(name)
                    resources.append(conversions.to_mcp_resource(loaded))
                except Exception:
                    logger.exception("Failed to load resource %r", name)
                    continue

            logger.debug("Found %d resources", len(resources))
            return resources

        @self.server.read_resource()
        async def handle_read_resource(uri: mcp.types.AnyUrl) -> str | bytes:
            """Handle direct resource content requests."""
            try:
                # Convert MCP URI to our format and load
                internal_uri = conversions.from_mcp_uri(str(uri))
                logger.debug("Loading resource from internal URI: %s", internal_uri)

                # For plain resource names, load by name first
                if "://" not in internal_uri:
                    resource = await self.resource_registry.load(internal_uri)
                else:
                    resource = await self.resource_registry.load_by_uri(internal_uri)

                # Handle different content types
                if resource.metadata.mime_type.startswith("text/"):
                    return resource.content
                # For binary content (like images), return raw bytes
                return resource.content_items[0].content.encode()

            except Exception as exc:
                error_msg = f"Failed to read resource: {exc}"
                logger.exception(error_msg)
                # Use proper MCP error response
                error = mcp.McpError(error_msg)
                error.error = mcp.ErrorData(code=INTERNAL_ERROR, message=str(exc))
                raise error from exc

        @self.server.completion()
        async def handle_completion(
            ref: mcp.types.PromptReference | mcp.types.ResourceReference,
            argument: mcp.types.CompletionArgument,
        ) -> CompleteResult:
            """Handle completion requests."""
            try:
                completion: Completion
                match ref:
                    case mcp.types.PromptReference():
                        completion = await self._complete_prompt_argument(
                            self.prompt_registry[ref.name],
                            argument.name,
                            argument.value,
                        )
                    case mcp.types.ResourceReference():
                        completion = await self._complete_resource(
                            AnyUrl(ref.uri), argument
                        )
                    case _:
                        msg = f"Invalid reference type: {type(ref)}"
                        error = mcp.McpError(msg)
                        error.error = mcp.ErrorData(code=INVALID_PARAMS, message=msg)
                        raise error  # noqa: TRY301

                return CompleteResult(completion=completion)

            except Exception as exc:
                logger.exception("Completion failed")
                error = mcp.McpError(msg)
                error.error = mcp.ErrorData(code=INTERNAL_ERROR, message=str(exc))
                raise error from exc

        @self.server.progress_notification()
        async def handle_progress(
            token: str | int,
            progress: float,
            total: float | None,
        ) -> None:
            """Handle progress notifications from client."""
            logger.debug(
                "Progress notification: %s %.1f/%.1f",
                token,
                progress,
                total or 0.0,
            )

    @classmethod
    def from_config_file(
        cls, config_path: str | os.PathLike[str], *, name: str = "llmling-server"
    ) -> LLMLingServer:
        """Create server from config file."""
        from llmling.config.loading import load_config

        return cls(load_config(config_path), name=name)

    async def start(self, *, raise_exceptions: bool = False) -> None:
        """Start the server."""
        try:
            # Initialize registries
            await self.processor_registry.startup()
            await self.tool_registry.startup()
            await self.resource_registry.startup()

            # Register tools from config
            for name, tool_config in self.config.tools.items():
                self.tool_registry[name] = tool_config

            # Register resources from config
            for name, resource in self.config.resources.items():
                self.resource_registry.register(name, resource)
            # Start MCP server
            handler = configure_server_logging(self.server)
            options = self.server.create_initialization_options()
            async with (
                mcp.stdio_server() as (read_stream, write_stream),
                asyncio.TaskGroup() as tg,
            ):
                tg.create_task(run_logging_processor(handler))
                # Run server
                await self.server.run(
                    read_stream,
                    write_stream,
                    options,
                    raise_exceptions=raise_exceptions,
                )
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Shutdown the server."""
        try:
            # Cancel all pending tasks
            if self._tasks:
                for task in self._tasks:
                    task.cancel()
                await asyncio.gather(*self._tasks, return_exceptions=True)

            # Remove observers
            if hasattr(self, "resource_observer"):
                await self.resource_observer.cleanup()
            if hasattr(self, "prompt_observer"):
                await self.prompt_observer.cleanup()
            if hasattr(self, "tool_observer"):
                await self.tool_observer.cleanup()
            self.resource_registry.remove_observer(self.resource_observer.events)
            self.prompt_registry.remove_observer(self.prompt_observer.events)
            self.tool_registry.remove_observer(self.tool_observer.events)

            # Shutdown registries
            await self.processor_registry.shutdown()
            await self.tool_registry.shutdown()
            await self.resource_registry.shutdown()

        finally:
            self._tasks.clear()

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Async context manager exit."""
        await self.shutdown()

    @property
    def current_session(self) -> mcp.ServerSession:
        """Get current session from request context."""
        try:
            return self.server.request_context.session
        except LookupError as exc:
            msg = "No active request context"
            raise RuntimeError(msg) from exc

    def notify_progress(
        self,
        token: str,
        progress: float,
        total: float | None = None,
        description: str | None = None,
    ) -> None:
        """Send progress notification to client."""
        try:
            # Get current session
            session = self.current_session

            # Create and track the progress notification task
            self._create_task(
                session.send_progress_notification(
                    progress_token=token,
                    progress=progress,
                    total=total,
                )
            )

            # Optionally send description as log message
            if description:
                self._create_task(
                    session.send_log_message(
                        level="info",
                        data=description,
                    )
                )

        except Exception:
            logger.exception("Failed to send progress notification")

    async def notify_resource_list_changed(self) -> None:
        """Notify clients about resource list changes."""
        try:
            await self.current_session.send_resource_list_changed()
        except RuntimeError:
            logger.debug("No active session for notification")
        except Exception:
            logger.exception("Failed to send resource list change notification")

    async def notify_resource_change(self, uri: str) -> None:
        """Notify clients about resource changes."""
        try:
            url = conversions.to_mcp_uri(uri)
            self._create_task(self.current_session.send_resource_updated(url))
            self._create_task(self.current_session.send_resource_list_changed())
        except RuntimeError:
            logger.debug("No active session for notification")
        except Exception:
            logger.exception("Failed to send resource change notification")

    async def notify_prompt_list_changed(self) -> None:
        """Notify clients about prompt list changes."""
        try:
            self._create_task(self.current_session.send_prompt_list_changed())
        except RuntimeError:
            logger.debug("No active session for notification")
        except Exception:
            logger.exception("Failed to send prompt list change notification")

    async def notify_tool_list_changed(self) -> None:
        """Notify clients about tool list changes."""
        try:
            self._create_task(self.current_session.send_tool_list_changed())
        except RuntimeError:
            logger.debug("No active session for notification")
        except Exception:
            logger.exception("Failed to send tool list change notification")

    async def _complete_prompt_argument(
        self,
        prompt: Prompt,
        arg_name: str,
        current_value: str,
    ) -> Completion:
        """Generate completions for prompt arguments."""
        # Find the argument definition
        arg_def = next(
            (arg for arg in prompt.arguments if arg.name == arg_name),
            None,
        )
        if not arg_def:
            return Completion(values=[], total=0, hasMore=False)

        values: list[str] = []
        match arg_def.type:
            case ArgumentType.ENUM:
                # Filter enum values based on current input
                values = [
                    v for v in (arg_def.enum_values or []) if v.startswith(current_value)
                ]

            case ArgumentType.RESOURCE:
                # Complete resource names
                values = [
                    name
                    for name in self.resource_registry.list_items()
                    if name.startswith(current_value)
                ]

            case ArgumentType.FILE:
                # Complete file paths
                import glob

                pattern = f"{current_value}*"
                values = list(glob.glob(pattern))  # noqa: PTH207

        return Completion(
            values=values[:100],  # Limit to 100 items
            total=len(values),
            hasMore=len(values) > 100,  # noqa: PLR2004
        )

    async def _complete_resource(
        self,
        uri: AnyUrl,
        argument: CompletionArgument,
    ) -> Completion:
        """Generate completions for resource fields."""
        try:
            # Convert AnyUrl to string for resource lookup
            str_uri = str(uri)
            resource = await self.resource_registry.load_by_uri(str_uri)
            values: list[str] = []

            # Different completion logic based on resource type
            match resource:
                case PathResource():
                    # Complete paths
                    import glob

                    pattern = f"{argument.value}*"
                    values = list(glob.glob(pattern))  # noqa: PTH207

                case SourceResource():
                    # Complete Python import paths
                    values = [
                        name
                        for name in self._get_importable_names()
                        if name.startswith(argument.value)
                    ]

            return Completion(
                values=values[:100],
                total=len(values),
                hasMore=len(values) > 100,  # noqa: PLR2004
            )

        except Exception:
            logger.exception("Resource completion failed")
            return Completion(values=[], total=0, hasMore=False)


async def serve(config_path: str | os.PathLike[str] | None = None) -> None:
    """Serve LLMling via MCP protocol.

    Args:
        config_path: Optional path to config file
    """
    logger = get_logger(__name__)

    # Create server instance
    server = LLMLingServer.from_config_file(config_path or config_resources.TEST_CONFIG)

    try:
        await server.start(raise_exceptions=True)
    except Exception:
        logger.exception("Server error")
        raise


if __name__ == "__main__":
    import asyncio
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else config_resources.TEST_CONFIG
    asyncio.run(serve(config_path))
