"""Logging-related protocol handlers."""

from __future__ import annotations

from mcp.types import (
    EmptyResult,
    LoggingLevel,
    ServerResult,
)

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.server.handlers.base import HandlerBase


logger = get_logger(__name__)


class LoggingHandlers(HandlerBase):
    """Logging protocol handlers."""

    def register(self) -> None:
        """Register logging handlers."""

        @self.server.server.set_logging_level()
        async def handle_set_logging_level(level: LoggingLevel) -> ServerResult:
            """Handle logging level changes."""
            try:
                logger.setLevel(level.upper())
                return ServerResult(root=EmptyResult())
            except Exception as exc:
                logger.exception("Failed to set logging level")
                msg = f"Failed to set logging level: {exc}"
                raise exceptions.ProcessorError(msg) from exc
