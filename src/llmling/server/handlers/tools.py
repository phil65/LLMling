"""Tool-related protocol handlers."""

from __future__ import annotations

from typing import Any

from mcp.types import (
    CallToolResult,
    ListToolsResult,
    ServerResult,
    Tool,
)

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.server.handlers.base import HandlerBase


logger = get_logger(__name__)


class ToolHandlers(HandlerBase):
    """Tool protocol handlers."""

    def register(self) -> None:
        """Register tool handlers."""

        @self.server.server.list_tools()
        async def handle_list_tools() -> ServerResult:
            """List available tools."""
            try:
                tools = []
                for tool in self.server.tool_registry.values():
                    schema = tool.get_schema()
                    tools.append(
                        Tool(
                            name=schema["function"]["name"],
                            description=schema["function"]["description"],
                            inputSchema=schema["function"]["parameters"],
                        )
                    )
                return ServerResult(root=ListToolsResult(tools=tools))
            except Exception as exc:
                logger.exception("Failed to list tools")
                raise exceptions.ProcessorError(f"Failed to list tools: {exc}") from exc

        @self.server.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict[str, Any] | None = None
        ) -> ServerResult:
            """Execute a tool."""
            try:
                # Track progress
                await self.server.notify_progress(
                    0.0, 1.0, description=f"Executing tool {name}"
                )

                # Execute tool
                result = await self.server.tool_registry.execute(
                    name, **(arguments or {})
                )

                # Complete progress
                await self.server.notify_progress(
                    1.0, 1.0, description="Tool execution completed"
                )

                # Convert result to MCP content
                content = await self.server.convert_to_mcp_content(result)
                return ServerResult(root=CallToolResult(content=content, isError=False))
            except Exception as exc:
                logger.exception("Tool execution failed")
                error_content = await self.server.convert_to_mcp_content(
                    f"Error executing tool: {exc}", error=True
                )
                return ServerResult(
                    root=CallToolResult(content=error_content, isError=True)
                )
