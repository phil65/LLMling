"""MCP protocol server implementation for LLMling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mcp
from mcp.server import Server as MCPServer
from mcp.server.models import InitializationOptions
from mcp.types import (
    AnyUrl,
    EmbeddedResource,
    ImageContent,
    Prompt,
    TextContent,
    Tool,
)

from llmling.core import exceptions
from llmling.core.log import get_logger


if TYPE_CHECKING:
    from llmling.config.models import Config
    from llmling.server.session import LLMLingSession

logger = get_logger(__name__)


class LLMLingMCPServer:
    """MCP server implementation for LLMling."""

    def __init__(
        self,
        name: str,
        config: Config,
        session: LLMLingSession,
    ) -> None:
        """Initialize server.

        Args:
            name: Server name
            config: Server configuration
            session: Active server session
        """
        self.name = name
        self.config = config
        self.session = session
        # Create MCP server with proper initialization options
        self.server = MCPServer(name)
        self.mcp_server = self.server
        # self.init_opts = self.server.create_initialization_options()
        self._setup_handlers()
        logger.debug("MCP server initialized with name: %s", name)

    def _setup_handlers(self) -> None:
        """Register MCP protocol handlers."""

        @self.mcp_server.read_resource()
        # @logfire.instrument("Reading resource: {uri}")
        async def handle_read_resource(uri: AnyUrl) -> str:
            """Read a specific resource."""
            try:
                # Find loader for URI
                loader = self.session.resource_registry.find_loader_for_uri(str(uri))

                # Track progress with proper type checking
                ctx = self.mcp_server.request_context
                meta = ctx.meta
                progress_token = meta.progressToken if meta is not None else None

                if progress_token is not None:
                    # Send progress notification
                    await ctx.session.send_progress_notification(
                        progress_token=progress_token,
                        progress=0,  # Integer
                        total=1,  # Integer
                    )
                    # Send description via log
                    await ctx.session.send_log_message(
                        level="info",
                        data=f"Loading resource {uri}",
                    )

                # Load resource
                result = await loader.load(
                    context=loader.context,
                    processor_registry=self.session.processor_registry,
                )
                session = self.mcp_server.request_context.session
                # Send completion progress
                if progress_token:
                    await session.send_progress_notification(
                        progress_token=progress_token,
                        progress=1,  # Use integer
                        total=1,  # Use integer
                    )
                    # Send completion via log message
                    await session.send_log_message(
                        level="info",
                        data="Resource loaded successfully",
                    )
            except Exception as exc:
                logger.exception("Failed to read resource")
                msg = f"Failed to read resource: {exc}"
                raise exceptions.ResourceError(msg) from exc
            else:
                return result.content

        @self.mcp_server.list_prompts()
        # @logfire.instrument("Listing available prompts")
        async def handle_list_prompts() -> list[Prompt]:
            """List available prompts."""
            logger.debug("Handling tools/list request")
            try:
                tools = []
                for tool in self.session.tool_registry.values():
                    logger.debug("Processing tool: %s", tool)
                    schema = tool.get_schema()
                    tools.append(
                        Tool(
                            name=schema["function"]["name"],
                            description=schema["function"]["description"],
                            inputSchema=schema["function"]["parameters"],
                        )
                    )
                logger.debug("Returning tools: %s", tools)
                return tools
            except Exception:
                logger.exception("Error listing tools")
                raise

        @self.mcp_server.get_prompt()
        async def handle_get_prompt(
            name: str, arguments: dict[str, str] | None = None
        ) -> mcp.types.GetPromptResult:
            """Get a specific prompt."""
            try:
                result = await self.session.prompt_registry.render(name, arguments or {})

                messages = []
                for msg in result.messages:
                    # Map internal role to MCP role (user/assistant only)
                    mcp_role: mcp.types.Role = (
                        "assistant" if msg.role == "assistant" else "user"
                    )

                    # Convert content
                    match msg.content:
                        case str():
                            mcp_content = mcp.types.TextContent(
                                type="text",
                                text=msg.content,
                            )
                        case list():
                            contents = []
                            for item in msg.get_content_items():
                                if item.type == "resource":
                                    # Find matching resolved content if any
                                    resolved_content = None
                                    if msg.resolved_content:
                                        for rc in msg.resolved_content:
                                            if rc.original == item and rc.resolved:
                                                resolved_content = rc.resolved
                                                break

                                    if resolved_content:
                                        contents.append(
                                            mcp.types.TextContent(
                                                type="text",
                                                text=resolved_content.content,
                                            )
                                        )
                                    else:
                                        contents.append(
                                            mcp.types.EmbeddedResource(
                                                type="resource",
                                                resource=mcp.types.TextResourceContents(
                                                    uri=mcp.types.AnyUrl(item.content),
                                                    text=item.alt_text or "",
                                                    mimeType="text/plain",
                                                ),
                                            )
                                        )
                                elif item.type in ("image_url", "image_base64"):
                                    contents.append(
                                        mcp.types.ImageContent(
                                            type="image",
                                            data=item.content,
                                            mimeType="image/png",
                                            # Or determine from content
                                        )
                                    )
                                else:  # text
                                    contents.append(
                                        mcp.types.TextContent(
                                            type="text",
                                            text=item.content,
                                        )
                                    )
                            # MCP expects a single content item, not a list
                            mcp_content = (
                                contents[0] if len(contents) == 1 else contents[0]
                            )

                    messages.append(
                        mcp.types.PromptMessage(
                            role=mcp_role,
                            content=mcp_content,
                        )
                    )

                return mcp.types.GetPromptResult(
                    description=f"Prompt: {name}",
                    messages=messages,
                )

            except Exception as exc:
                logger.exception("Failed to get prompt")
                error_msg = f"Failed to get prompt: {exc}"
                raise exceptions.ProcessorError(error_msg) from exc

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """Handle tools/list request."""
            try:
                tools = []
                for tool in self.session.tool_registry.values():
                    schema = tool.get_schema()
                    tools.append(
                        Tool(
                            name=schema["function"]["name"],
                            description=schema["function"]["description"],
                            inputSchema=schema["function"]["parameters"],
                        )
                    )
            except Exception:
                logger.exception("Error listing tools")
                raise
            else:
                return tools

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict[str, Any] | None = None
        ) -> list[TextContent | ImageContent | EmbeddedResource]:
            result = await self.session.tool_registry.execute(name, **(arguments or {}))
            return [TextContent(type="text", text=str(result))]

    def _convert_to_mcp_content(
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

        # Handle lists and tuples
        if isinstance(content, list | tuple):
            results = []
            for item in content:
                results.extend(self._convert_to_mcp_content(item))
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

    async def start(self, *, raise_exceptions: bool = False) -> None:
        """Start MCP server."""
        try:
            await self.session.startup()

            # Create proper initialization options
            init_options = InitializationOptions(
                server_name=self.name,
                server_version="1.0.0",
                capabilities=mcp.types.ServerCapabilities(
                    tools=mcp.types.ToolsCapability(listChanged=True),
                    prompts=mcp.types.PromptsCapability(listChanged=True),
                    resources=mcp.types.ResourcesCapability(
                        subscribe=True,
                        listChanged=True,
                    ),
                    logging=mcp.types.LoggingCapability(),
                ),
            )

            # Run server with initialization options
            async with mcp.stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    initialization_options=init_options,
                    raise_exceptions=raise_exceptions,
                )
        finally:
            await self.session.close()
