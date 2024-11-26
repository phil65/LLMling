"""Base server functionality for LLMling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from mcp.server import Server as MCPServer
from mcp.types import (
    EmbeddedResource,
    ImageContent,
    LoggingLevel,
    ServerResult,
    TextContent,
)

from llmling.core import exceptions
from llmling.core.log import get_logger


if TYPE_CHECKING:
    from mcp.server.session import ServerSession
    from mcp.shared.context import RequestContext

logger = get_logger(__name__)


class ServerBase:
    """Base class for LLMling MCP server implementation."""

    def __init__(self, name: str) -> None:
        """Initialize base server.

        Args:
            name: Server name
        """
        self.name = name
        self.server = MCPServer(name)

    @property
    def request_context(self) -> RequestContext[ServerSession]:
        """Get current request context.

        Returns:
            Current request context

        Raises:
            RuntimeError: If called outside request context
        """
        try:
            return cast(RequestContext[ServerSession], self.server.request_context)
        except LookupError as exc:
            msg = "No active request context"
            raise RuntimeError(msg) from exc

    async def notify_progress(
        self,
        progress: float,
        total: float | None = None,
        *,
        description: str | None = None,
    ) -> None:
        """Send progress notification if progress token exists.

        Args:
            progress: Current progress value
            total: Optional total value
            description: Optional progress description
        """
        try:
            ctx = self.request_context
            if progress_token := ctx.meta.progressToken:
                # Send progress notification
                await ctx.session.send_progress_notification(
                    progress_token=progress_token,
                    progress=progress,
                    total=total,
                )
                # Send description as log message if provided
                if description:
                    await ctx.session.send_log_message(
                        level="info",
                        data=description,
                    )
        except Exception:
            logger.warning("Failed to send progress notification", exc_info=True)

    async def convert_to_mcp_content(
        self,
        content: Any,
        *,
        error: bool = False,
    ) -> list[TextContent | ImageContent | EmbeddedResource]:
        """Convert content to MCP content types.

        Args:
            content: Content to convert
            error: Whether this is error content

        Returns:
            List of MCP content types
        """
        # Handle None
        if content is None:
            return [TextContent(type="text", text="")]

        # Handle strings
        if isinstance(content, str):
            return [TextContent(type="text", text=content)]

        # Handle bytes (convert to base64)
        if isinstance(content, bytes):
            import base64

            data = base64.b64encode(content).decode()
            return [
                TextContent(
                    type="text", text=f"Binary data ({len(content)} bytes): {data[:100]}"
                )
            ]

        # Handle lists and tuples
        if isinstance(content, (list, tuple)):
            results = []
            for item in content:
                results.extend(await self.convert_to_mcp_content(item))
            return results

        # Handle dictionaries
        if isinstance(content, dict):
            import json

            return [TextContent(type="text", text=json.dumps(content, indent=2))]

        # Handle exceptions
        if isinstance(content, Exception):
            return [TextContent(type="text", text=f"Error: {content!s}")]

        # Handle everything else
        return [TextContent(type="text", text=str(content))]

    async def wrap_result(
        self,
        result: Any,
        *,
        error: bool = False,
    ) -> ServerResult:
        """Wrap result in proper MCP response type.

        Args:
            result: Result to wrap
            error: Whether this is an error result

        Returns:
            Wrapped MCP result
        """
        # If already wrapped, return as-is
        if isinstance(result, ServerResult):
            return result

        try:
            # Convert to MCP content types
            content = await self.convert_to_mcp_content(result, error=error)

            # Return as CallToolResult for tools, TextContent for others
            if self.request_context.request.method == "tools/call":
                from mcp.types import CallToolResult

                return ServerResult(
                    root=CallToolResult(
                        content=content,
                        isError=error,
                    )
                )
            return ServerResult(
                root=content[0] if content else TextContent(type="text", text="")
            )

        except Exception as exc:
            logger.exception("Failed to wrap result")
            raise exceptions.ProcessorError(f"Failed to process result: {exc}") from exc

    async def log_message(
        self,
        level: LoggingLevel,
        message: str,
        *,
        logger_name: str | None = None,
    ) -> None:
        """Send log message to client.

        Args:
            level: Log level
            message: Log message
            logger_name: Optional logger name
        """
        try:
            await self.request_context.session.send_log_message(
                level=level,
                data=message,
                logger=logger_name,
            )
        except Exception:
            logger.warning("Failed to send log message", exc_info=True)

    async def notify_resource_change(self, uri: str) -> None:
        """Notify clients about resource changes.

        Args:
            uri: Resource URI that changed
        """
        try:
            await self.request_context.session.send_resource_updated(uri)
            await self.request_context.session.send_resource_list_changed()
        except Exception:
            logger.warning("Failed to send resource change notification", exc_info=True)

    async def notify_prompt_change(self) -> None:
        """Notify clients about prompt changes."""
        try:
            await self.request_context.session.send_prompt_list_changed()
        except Exception:
            logger.warning("Failed to send prompt change notification", exc_info=True)

    async def notify_tool_change(self) -> None:
        """Notify clients about tool changes."""
        try:
            await self.request_context.session.send_tool_list_changed()
        except Exception:
            logger.warning("Failed to send tool change notification", exc_info=True)
