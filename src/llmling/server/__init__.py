"""MCP protocol server implementation for LLMling."""

import os
from llmling.server.factory import create_server
from llmling.server.server import LLMLingServer


async def serve(config_path: str | os.PathLike[str] | None = None) -> None:
    """Serve LLMling via MCP protocol.

    Args:
        config_path: Optional path to config file

    Raises:
        ConfigError: If configuration is invalid
        Exception: If server fails to start
    """
    server = create_server(config_path)
    await server.start(raise_exceptions=True)


__all__ = [
    "LLMLingServer",
    "create_server",
    "serve",
]
