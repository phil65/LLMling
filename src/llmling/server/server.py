"""MCP server implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import mcp
from mcp.server import Server
from mcp.types import GetPromptResult, TextContent
from pydantic import AnyUrl

from llmling.core.log import get_logger
from llmling.processors.registry import ProcessorRegistry
from llmling.prompts.registry import PromptRegistry
from llmling.resources import ResourceLoaderRegistry
from llmling.resources.registry import ResourceRegistry
from llmling.server import conversions
from llmling.server.observers import PromptObserver, ResourceObserver, ToolObserver
from llmling.tools.registry import ToolRegistry


if TYPE_CHECKING:
    from llmling.config.models import Config

logger = get_logger(__name__)


class LLMLingServer:
    def __init__(
        self,
        config: Config,
        *,
        name: str = "llmling-server",
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

    @classmethod
    def from_config_file(
        cls, config_path: str, *, name: str = "llmling-server"
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
            options = self.server.create_initialization_options()
            async with mcp.stdio_server() as (read_stream, write_stream):
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
            # Handle different URI types according to MCP spec
            if uri.startswith(("http://", "https://")):
                # Pass through remote URLs that clients can access directly
                mcp_uri = uri
            elif uri.startswith("file:"):
                # Ensure proper file:/// format for MCP
                path = uri.split(":", 1)[1].lstrip("/")
                mcp_uri = f"file:///{path}"
            else:
                logger.warning("Unsupported URI scheme: %s", uri)
                return

            url = AnyUrl(mcp_uri)
            await self.current_session.send_resource_updated(url)
        except RuntimeError:
            logger.debug("No active session for notification")
        except Exception:
            logger.exception("Failed to send resource change notification")

    async def notify_prompt_list_changed(self) -> None:
        """Notify clients about prompt list changes."""
        try:
            await self.current_session.send_prompt_list_changed()
        except RuntimeError:
            logger.debug("No active session for notification")
        except Exception:
            logger.exception("Failed to send prompt list change notification")

    async def notify_tool_list_changed(self) -> None:
        """Notify clients about tool list changes."""
        try:
            await self.current_session.send_tool_list_changed()
        except RuntimeError:
            logger.debug("No active session for notification")
        except Exception:
            logger.exception("Failed to send tool list change notification")


async def serve(config_path: str | None = None) -> None:
    """Serve LLMling via MCP protocol.

    Args:
        config_path: Optional path to config file
    """
    logger = get_logger(__name__)

    # Create server instance
    server = LLMLingServer.from_config_file(
        config_path or "src/llmling/config_resources/test.yml"
    )

    try:
        await server.start(raise_exceptions=True)
    except Exception:
        logger.exception("Server error")
        raise


if __name__ == "__main__":
    import asyncio
    import sys

    config_path = (
        sys.argv[1] if len(sys.argv) > 1 else "src/llmling/config_resources/test.yml"
    )
    asyncio.run(serve(config_path))
