from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
import mcp
from mcp.server.models import InitializationOptions
from mcp.types import (
    JSONRPCMessage,
    ServerCapabilities,
    ResourcesCapability,
    PromptsCapability,
    ToolsCapability,
    LoggingCapability,
)


from llmling.core.log import get_logger
from llmling.processors.registry import ProcessorRegistry
from llmling.prompts.registry import PromptRegistry
from llmling.resources import ResourceLoaderRegistry, default_registry
from llmling.tools.registry import ToolRegistry
from llmling.server.base import ServerBase
from llmling.server.handlers import (
    ResourceHandlers,
    PromptHandlers,
    ToolHandlers,
    LoggingHandlers,
)

if TYPE_CHECKING:
    from llmling.config.models import Config

logger = get_logger(__name__)

__all__ = ["LLMLingServer", "create_server"]


class LLMLingServer(ServerBase):
    """MCP server implementation for LLMling."""

    def __init__(
        self,
        config: Config,
        *,
        resource_registry: ResourceLoaderRegistry | None = None,
        processor_registry: ProcessorRegistry | None = None,
        prompt_registry: PromptRegistry | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        """Initialize server with registries.

        Args:
            config: Server configuration
            resource_registry: Optional custom resource registry
            processor_registry: Optional custom processor registry
            prompt_registry: Optional custom prompt registry
            tool_registry: Optional custom tool registry
        """
        super().__init__("llmling")

        # Store configuration and registries
        self.config = config
        self.resource_registry = resource_registry or default_registry
        self.processor_registry = processor_registry or ProcessorRegistry()
        self.prompt_registry = prompt_registry or PromptRegistry()
        self.tool_registry = tool_registry or ToolRegistry()

        # Initialize handlers
        self._init_handlers()

    def _init_handlers(self) -> None:
        """Initialize all protocol handlers."""
        handlers = [
            ResourceHandlers(self),
            PromptHandlers(self),
            ToolHandlers(self),
            LoggingHandlers(self),
        ]
        for handler in handlers:
            handler.register()

    def _get_capabilities(self) -> ServerCapabilities:
        """Get server capabilities."""
        return ServerCapabilities(
            resources=ResourcesCapability(
                subscribe=True,  # Enable resource subscriptions
                listChanged=True,  # Enable resource list change notifications
            ),
            prompts=PromptsCapability(
                listChanged=True,  # Enable prompt list change notifications
            ),
            tools=ToolsCapability(
                listChanged=True,  # Enable tool list change notifications
            ),
            logging=LoggingCapability(),  # Enable logging capabilities
        )

    async def start(
        self,
        read_stream: MemoryObjectReceiveStream[JSONRPCMessage] | None = None,
        write_stream: MemoryObjectSendStream[JSONRPCMessage] | None = None,
    ) -> None:
        """Start the server.

        Args:
            read_stream: Optional read stream for testing
            write_stream: Optional write stream for testing
        """
        if read_stream is None or write_stream is None:
            # Use stdio by default
            async with mcp.stdio_server.stdio.stdio_server() as (r, w):
                await self.server.run(
                    r,
                    w,
                    self.server.create_initialization_options(),
                )
        else:
            # Use provided streams (for testing)
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


async def create_server(config_path: str) -> LLMLingServer:
    """Create and start MCP server from config.

    Args:
        config_path: Path to configuration file

    Returns:
        Initialized server instance
    """
    from llmling.config.loading import load_config

    # Load config
    config = load_config(config_path)

    # Create server
    server = LLMLingServer(config)

    return server


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    from llmling.testing.processors import uppercase_text, multiply
    from llmling.testing.tools import analyze_ast, example_tool

    async def test_server() -> None:
        """Test server functionality."""
        try:
            sys.stdout.reconfigure(line_buffering=True)  # type: ignore
            logger.info("Initializing test server...")

            config_path = Path(__file__).parent.parent / "config_resources" / "test.yml"
            logger.info("Loading config from: %s", config_path)

            server = await create_server(str(config_path))

            # Register test components
            server.processor_registry.register("uppercase", uppercase_text)
            server.processor_registry.register("multiply", multiply)
            server.tool_registry.register("analyze", analyze_ast)
            server.tool_registry.register("example", example_tool)

            logger.info("Starting server...")
            await server.start()

        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception:
            logger.exception("Fatal error")
            sys.exit(1)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    try:
        asyncio.run(test_server())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)
