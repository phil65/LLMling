from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    PromptMessage,
    Prompt,
    PromptArgument,
    GetPromptResult,
)

from llmling.core.log import get_logger
from llmling.processors.registry import ProcessorRegistry
from llmling.prompts.registry import PromptRegistry
from llmling.resources import ResourceLoaderRegistry, default_registry
from llmling.tools.registry import ToolRegistry
from pydantic import AnyUrl

if TYPE_CHECKING:
    from llmling.config.models import Config

logger = get_logger(__name__)


class LLMLingServer:
    """MCP server implementation for LLMLing."""

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
        self.config = config
        self.resource_registry = resource_registry or default_registry
        self.processor_registry = processor_registry or ProcessorRegistry()
        self.prompt_registry = prompt_registry or PromptRegistry()
        self.tool_registry = tool_registry or ToolRegistry()

        # Create MCP server
        self.server = Server("llmling")
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Set up protocol handlers."""

        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            """List available resources."""
            resources = []

            # Convert our contexts to MCP resources
            for name, context in self.config.contexts.items():
                # Get loader class for context type
                loader_class = self.resource_registry[context.context_type]
                # Create loader instance with context
                loader = loader_class.create(context)
                uri = loader.create_uri(name=name)

                # TODO: Get mimetype from loader.supported_mime_types[0]
                resources.append(
                    Resource(
                        uri=AnyUrl(uri),
                        name=name,
                        description=context.description,
                        mimeType="text/plain",
                    )
                )

            return resources

        @self.server.read_resource()
        async def handle_read_resource(uri: AnyUrl) -> str:
            """Read a specific resource."""
            try:
                # Convert AnyUrl to str for internal use
                uri_str = str(uri)
                loader = self.resource_registry.find_loader_for_uri(uri_str)
                result = await loader.load(
                    context=loader.context, processor_registry=self.processor_registry
                )
            except Exception:
                logger.exception("Failed to read resource %s", uri)
                raise
            else:
                return result.content

        @self.server.list_prompts()
        async def handle_list_prompts() -> list[Prompt]:
            """List available prompts."""
            return [
                Prompt(
                    name=prompt.name,
                    description=prompt.description,
                    arguments=[
                        PromptArgument(
                            name=arg.name,
                            description=arg.description,
                            required=arg.required,
                        )
                        for arg in prompt.arguments
                    ],
                )
                for prompt in self.prompt_registry.values()
            ]

        @self.server.get_prompt()
        async def handle_get_prompt(
            name: str, arguments: dict[str, str] | None = None
        ) -> GetPromptResult:
            """Get a specific prompt."""
            result = await self.prompt_registry.render(name, arguments or {})

            return GetPromptResult(
                description=f"Prompt: {name}",
                messages=[
                    PromptMessage(
                        role=msg.role, content=TextContent(type="text", text=msg.content)
                    )
                    for msg in result.messages
                ],
            )

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available tools."""
            tools = []

            for tool in self.tool_registry.values():
                schema = tool.get_schema()
                tools.append(
                    Tool(
                        name=schema["function"]["name"],
                        description=schema["function"]["description"],
                        inputSchema=schema["function"]["parameters"],
                    )
                )

            return tools

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict[str, Any] | None = None
        ) -> list[TextContent | ImageContent | EmbeddedResource]:
            """Execute a tool."""
            try:
                result = await self.tool_registry.execute(name, **(arguments or {}))

                # Convert result to MCP content type
                if isinstance(result, str):
                    return [TextContent(type="text", text=result)]
                if isinstance(result, bytes):
                    # Handle binary data
                    return [
                        TextContent(
                            type="text", text=f"Binary data ({len(result)} bytes)"
                        )
                    ]
                if isinstance(result, dict):
                    return [TextContent(type="text", text=str(result))]
                return [TextContent(type="text", text=str(result))]

            except Exception:
                logger.exception("Tool execution failed")
                raise

    async def start(self) -> None:
        """Start the server with stdio transport."""
        import mcp.server.stdio

        # Initialize registries
        await self.processor_registry.startup()

        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="llmling",
                    server_version="0.4.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


# Helper function to create and run server
async def create_server(config_path: str) -> None:
    """Create and start MCP server from config."""
    from llmling.config.loading import load_config

    # Load config
    config = load_config(config_path)

    # Create server
    server = LLMLingServer(config)

    # Start server
    await server.start()


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    from llmling.config.loading import load_config
    from llmling.testing.processors import uppercase_text, multiply
    from llmling.testing.tools import analyze_ast, example_tool

    async def test_server():
        """Test server functionality with example config."""
        try:
            sys.stdout.reconfigure(line_buffering=True)  # type: ignore
            print("\nInitializing test server...", flush=True)

            config_path = Path(__file__).parent.parent / "config_resources" / "test.yml"
            print(f"Loading config from: {config_path}", flush=True)
            config = load_config(config_path)

            server = LLMLingServer(config)

            # Register test components
            server.processor_registry.register("uppercase", uppercase_text)
            server.processor_registry.register("multiply", multiply)
            server.tool_registry.register("analyze", analyze_ast)
            server.tool_registry.register("example", example_tool)

            print("\nServer capabilities:", flush=True)
            print("------------------", flush=True)
            caps = server.server.get_capabilities(NotificationOptions(), {})
            # Access capabilities directly as attributes
            print(f"Resources: {bool(caps.resources)}", flush=True)
            print(f"Tools: {bool(caps.tools)}", flush=True)
            print(f"Prompts: {bool(caps.prompts)}", flush=True)

            print("\nLoaded configuration:", flush=True)
            print(f"Contexts: {len(config.contexts)}", flush=True)
            print(f"Tools: {len(config.tools)}", flush=True)

            print("\nStarting MCP server (Ctrl+C to exit)...", flush=True)
            print("-" * 40, flush=True)
            sys.stdout.flush()
            sys.stderr.flush()

            # Create stdio transport
            import mcp.server.stdio

            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await server.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="llmling",
                        server_version="0.4.0",
                        capabilities=server.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    ),
                )

        except KeyboardInterrupt:
            print("\nServer stopped by user", flush=True)
            return
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr, flush=True)
            raise

    # Set up logging before we start
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    try:
        asyncio.run(test_server())
    except KeyboardInterrupt:
        print("\nShutdown complete", flush=True)
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        print(f"Fatal error: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
