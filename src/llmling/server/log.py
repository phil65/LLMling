"""Server-specific logging configuration."""

from __future__ import annotations

import asyncio
import logging
import queue
from typing import TYPE_CHECKING, Any

from llmling.core.log import setup_logging


if TYPE_CHECKING:
    from mcp import types
    from mcp.server import Server


# Map Python logging levels to MCP logging levels
LEVEL_MAP: dict[int, types.LoggingLevel] = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warning",
    logging.ERROR: "error",
    logging.CRITICAL: "critical",
}


class MCPHandler(logging.Handler):
    """Handler that sends logs via MCP protocol."""

    def __init__(self, mcp_server: Server) -> None:
        """Initialize handler with MCP server instance."""
        super().__init__()
        self.server = mcp_server
        self.queue: queue.Queue[tuple[types.LoggingLevel, Any, str | None]] = (
            queue.Queue()
        )

    def emit(self, record: logging.LogRecord) -> None:
        """Queue log message for async sending."""
        try:
            # Try to get current session from server's request context
            try:
                _ = self.server.request_context  # Check if we have a context
            except LookupError:
                # No active session - only use file logging
                return

            # Convert Python logging level to MCP level
            level = LEVEL_MAP.get(record.levelno, "info")

            # Format message
            message: Any = self.format(record)

            # Queue for async processing
            self.queue.put((level, message, record.name))

        except Exception:  # noqa: BLE001
            self.handleError(record)

    async def process_queue(self) -> None:
        """Process queued log messages."""
        while True:
            try:
                # Get session for each message (might have changed)
                session = self.server.request_context.session

                # Process all available messages
                while not self.queue.empty():
                    level, data, logger_name = self.queue.get_nowait()
                    await session.send_log_message(
                        level=level,
                        data=data,
                        logger=logger_name,
                    )
                    self.queue.task_done()

            except LookupError:
                # No active session - messages will stay in queue
                pass
            except Exception:
                # Log processing error to file only
                logger = logging.getLogger(__name__)
                logger.exception("Error processing log messages")

            # Wait before next attempt
            await asyncio.sleep(0.1)


def configure_server_logging(mcp_server: Server) -> MCPHandler:
    """Configure server logging to use file and MCP protocol (no stdout).

    Args:
        mcp_server: The MCP server instance to use for logging

    Returns:
        The configured MCP handler for queue processing
    """
    # Setup core logging in server mode (no stdout)
    setup_logging(
        level=logging.INFO,
        mode="server",  # This ensures no stdout handlers are created
    )

    # Add MCP handler
    mcp_handler = MCPHandler(mcp_server)
    mcp_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    # Add to root logger
    root = logging.getLogger()
    root.addHandler(mcp_handler)

    return mcp_handler


async def run_logging_processor(handler: MCPHandler) -> None:
    """Run the logging processor."""
    await handler.process_queue()
