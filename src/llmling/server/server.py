"""Main server implementation for LLMling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import logfire

from llmling.core.log import get_logger
from llmling.processors.registry import ProcessorRegistry
from llmling.prompts.registry import PromptRegistry
from llmling.resources import (
    default_registry as default_resource_registry,
)
from llmling.server.mcp_server import LLMLingMCPServer
from llmling.server.session import LLMLingSession
from llmling.tools.registry import ToolRegistry


if TYPE_CHECKING:
    import types

    from llmling.config.models import Config

logger = get_logger(__name__)


class LLMLingServer:
    """Main server class."""

    def __init__(
        self,
        config: Config,
        name: str = "llmling-server",
    ) -> None:
        """Initialize server."""
        self.config = config
        self.name = name
        self.session = LLMLingSession(
            config=config,
            resource_registry=default_resource_registry,
            processor_registry=ProcessorRegistry(),
            prompt_registry=PromptRegistry(),
            tool_registry=ToolRegistry(),
        )
        self.mcp_server = LLMLingMCPServer(name, config, self.session)

    @property
    def registries(self) -> dict[str, Any]:
        """Get all active registries."""
        return {
            "resources": self.session.resource_registry,
            "processors": self.session.processor_registry,
            "prompts": self.session.prompt_registry,
            "tools": self.session.tool_registry,
        }

    @logfire.instrument("Starting server")
    async def start(self, *, raise_exceptions: bool = False) -> None:
        """Start the server."""
        try:
            # Let MCP server handle the startup exactly like reference
            await self.mcp_server.start(raise_exceptions=raise_exceptions)
        except Exception as exc:
            logger.exception("Server startup failed")
            await self.shutdown()
            msg = "Failed to start server"
            raise RuntimeError(msg) from exc

    @logfire.instrument("Shutting down server")
    async def shutdown(self) -> None:
        """Shutdown the server."""
        await self.session.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        try:
            await self.shutdown()
        except Exception:
            logger.exception("Error during cleanup")
            raise


def create_server(config_path: str) -> LLMLingServer:
    """Create and configure a server instance.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured server instance

    Raises:
        ConfigError: If configuration loading fails
    """
    from llmling.config.loading import load_config

    # Load configuration
    config = load_config(config_path)

    # Create server with default registries
    return LLMLingServer(config)


if __name__ == "__main__":
    import asyncio
    import sys

    async def run_server(config_path: str) -> None:
        """Run server with configuration."""
        try:
            server = create_server(config_path)
            async with server:
                await server.start()
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception:
            logger.exception("Fatal server error")
            sys.exit(1)

    config_path = "src/llmling/config_resources/test.yml"
    asyncio.run(run_server(str(config_path)))
