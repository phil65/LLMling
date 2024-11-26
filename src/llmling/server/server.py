"""MCP server implementation."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any, Self

import mcp
from mcp.server import Server
from mcp.types import GetPromptResult, TextContent

from llmling.core.log import get_logger
from llmling.processors.registry import ProcessorRegistry
from llmling.prompts.registry import PromptRegistry
from llmling.resources import ResourceLoaderRegistry
from llmling.server import conversions
from llmling.tools.registry import ToolRegistry


if TYPE_CHECKING:
    from llmling.config.models import Config

logger = get_logger(__name__)


class LLMLingServer:
    """MCP server implementation."""

    def __init__(
        self,
        config: Config,
        *,
        name: str = "llmling-server",
        resource_registry: ResourceLoaderRegistry | None = None,
        processor_registry: ProcessorRegistry | None = None,
        prompt_registry: PromptRegistry | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        """Initialize server.

        Args:
            config: Server configuration
            name: Server name
            resource_registry: Optional resource registry
            processor_registry: Optional processor registry
            prompt_registry: Optional prompt registry
            tool_registry: Optional tool registry
        """
        self.config = config
        self.name = name

        # Initialize registries
        self.resource_registry = resource_registry or ResourceLoaderRegistry()
        self.processor_registry = processor_registry or ProcessorRegistry()
        self.prompt_registry = prompt_registry or PromptRegistry()
        self.tool_registry = tool_registry or ToolRegistry()

        # Create MCP server
        self.server = Server(name)
        self._setup_handlers()
        logger.debug("Server initialized with name: %s", name)

    @classmethod
    def from_config_file(
        cls, config_path: str, *, name: str = "llmling-server"
    ) -> LLMLingServer:
        """Create server from config file.

        Args:
            config_path: Path to configuration file
            name: Optional server name

        Returns:
            Configured server instance
        """
        from llmling.config.loading import load_config

        return cls(load_config(config_path), name=name)

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
            except Exception as exc:  # noqa: BLE001
                return [TextContent(type="text", text=str(exc))]

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
            return GetPromptResult(
                description=f"Prompt: {name}",
                messages=[conversions.to_mcp_message(msg) for msg in result.messages],
            )

        @self.server.list_resources()
        async def handle_list_resources() -> list[mcp.types.Resource]:
            """Handle resources/list request."""
            resources = []
            for context in self.config.contexts.values():
                loader = self.resource_registry[context.context_type]
                result = await loader.load(
                    context=context,
                    processor_registry=self.processor_registry,
                )
                resources.append(conversions.to_mcp_resource(result))
            return resources

        @self.server.read_resource()
        async def handle_read_resource(uri: mcp.types.AnyUrl) -> str:
            """Handle read_resource request."""
            loader = self.resource_registry.find_loader_for_uri(str(uri))
            result = await loader.load(
                context=loader.context,
                processor_registry=self.processor_registry,
            )
            return result.content

    async def start(self, *, raise_exceptions: bool = False) -> None:
        """Start the server."""
        try:
            # Initialize registries
            await self.processor_registry.startup()
            await self.tool_registry.startup()

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
        await self.processor_registry.shutdown()
        await self.tool_registry.shutdown()

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Async context manager exit."""
        await self.shutdown()


if __name__ == "__main__":
    config_path = (
        sys.argv[1] if len(sys.argv) > 1 else "src/llmling/config_resources/test.yml"
    )

    # Create and run server
    server = LLMLingServer.from_config_file(config_path)
    asyncio.run(server.start(raise_exceptions=True))
