"""SSE (Server-Sent Events) transport implementation for LLMling."""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route
import uvicorn

from llmling.core.log import get_logger
from llmling.server.log import configure_server_logging, run_logging_processor
from llmling.server.transports.base import TransportBase


logger = get_logger(__name__)


class SSEServer(TransportBase):
    """SSE transport implementation for LLMling server.

    This transport enables web clients to connect to the LLMling server using
    Server-Sent Events (SSE) for server-to-client messages and HTTP POST for
    client-to-server messages.

    Args:
        server: The MCP server instance
        host: Host address to bind to (default: "localhost")
        port: Port number to listen on (default: 8000)
        cors_origins: List of allowed CORS origins (default: ["*"])

    Example:
        ```python
        mcp_server = Server("example-server")
        sse = SSEServer(
            mcp_server,
            host="0.0.0.0",
            port=8000,
            cors_origins=["https://example.com"]
        )
        await sse.serve()
        ```
    """

    def __init__(
        self,
        server: Server,
        *,
        host: str = "localhost",
        port: int = 8000,
        cors_origins: list[str] | None = None,
    ) -> None:
        super().__init__(server)
        self.host = host
        self.port = port
        self.cors_origins = cors_origins or ["*"]
        self._app: Starlette | None = None
        self._server: uvicorn.Server | None = None

    def _create_app(self, raise_exceptions: bool) -> Starlette:
        """Create the Starlette ASGI application.

        Args:
            raise_exceptions: Whether to raise exceptions for debugging

        Returns:
            Configured Starlette application
        """
        # Create SSE transport
        sse = SseServerTransport("/messages")
        handler = configure_server_logging(self.server)

        async def handle_sse(scope: dict, receive: Any, send: Any) -> None:
            """Handle SSE connection endpoint."""
            logger.info(
                "New SSE connection from %s", scope.get("client", ("unknown", 0))[0]
            )

            async with (
                sse.connect_sse(scope, receive, send) as streams,
                asyncio.TaskGroup() as tg,
            ):
                # Start log processor for this connection
                tg.create_task(run_logging_processor(handler))

                try:
                    # Run MCP server for this connection
                    await self.server.run(
                        streams[0],
                        streams[1],
                        self.server.create_initialization_options(),
                        raise_exceptions=raise_exceptions,
                    )
                except Exception:
                    logger.exception("Connection error")
                    if raise_exceptions:
                        raise
                finally:
                    logger.info("SSE connection closed")

        async def handle_messages(scope: dict, receive: Any, send: Any) -> None:
            """Handle POST messages from clients."""
            try:
                await sse.handle_post_message(scope, receive, send)
            except Exception:
                logger.exception("Error handling message")
                if raise_exceptions:
                    raise

        # Create Starlette app with routes
        app = Starlette(
            debug=raise_exceptions,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Route("/messages", endpoint=handle_messages, methods=["POST"]),
            ],
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

        self._app = app
        return app

    async def serve(self, *, raise_exceptions: bool = False) -> None:
        """Start serving the SSE transport.

        Args:
            raise_exceptions: Whether to raise exceptions instead of handling them

        Raises:
            RuntimeError: If server fails to start
        """
        try:
            app = self._create_app(raise_exceptions)
            config = uvicorn.Config(
                app,
                host=self.host,
                port=self.port,
                log_level="debug" if raise_exceptions else "info",
                access_log=raise_exceptions,
            )
            server = uvicorn.Server(config)
            self._server = server
            msg = "Starting SSE server on %s:%d (CORS: %s)"
            logger.info(msg, self.host, self.port, ", ".join(self.cors_origins))

            # Handle graceful shutdown
            async def shutdown_handler() -> None:
                logger.info("Received shutdown signal")
                if self._server:
                    self._server.should_exit = True

            try:
                await server.serve()
            except Exception as exc:
                msg = "Failed to start SSE server"
                raise RuntimeError(msg) from exc

        except Exception:
            logger.exception("SSE server error")
            raise
        finally:
            self._server = None
            logger.info("SSE server stopped")

    async def shutdown(self) -> None:
        """Gracefully shutdown the SSE server."""
        if self._server:
            self._server.should_exit = True
            logger.info("SSE server shutdown initiated")


if __name__ == "__main__":
    # Example usage
    async def main() -> None:
        server = Server("example-server")
        transport = SSEServer(
            server, host="localhost", port=8000, cors_origins=["http://localhost:3000"]
        )
        await transport.serve(raise_exceptions=True)

    asyncio.run(main())
