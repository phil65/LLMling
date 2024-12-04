from __future__ import annotations

import asyncio

from mcp.server.stdio import stdio_server

from llmling.core.log import get_logger
from llmling.server.log import configure_server_logging, run_logging_processor
from llmling.server.transports.base import TransportBase


logger = get_logger(__name__)


class StdioServer(TransportBase):
    """Stdio transport implementation."""

    async def serve(self, *, raise_exceptions: bool = False) -> None:
        """Start the stdio server."""
        handler = configure_server_logging(self.server)

        async with stdio_server() as (read_stream, write_stream):
            logger.info("Starting stdio server")
            async with asyncio.TaskGroup() as tg:
                tg.create_task(run_logging_processor(handler))
                try:
                    await self.server.run(
                        read_stream,
                        write_stream,
                        self.server.create_initialization_options(),
                        raise_exceptions=raise_exceptions,
                    )
                except Exception:
                    logger.exception("Server error")
                    if raise_exceptions:
                        raise

    async def shutdown(self) -> None:
        """Cleanup stdio transport."""
        logger.info("Stdio transport shutdown")
