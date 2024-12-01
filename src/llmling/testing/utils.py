from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, TypeVar, cast

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.server.models import InitializationOptions
from mcp.server.session import ServerSession
from mcp.types import (
    JSONRPCMessage,
    LoggingCapability,
    PromptsCapability,
    ResourcesCapability,
    ServerCapabilities,
    ToolsCapability,
)


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


T_Stream = TypeVar("T_Stream")


class TestStreamPair[T_Stream]:
    """Test stream implementation."""

    def __init__(self) -> None:
        """Initialize stream pair."""
        # Create streams
        streams = anyio.create_memory_object_stream[T_Stream](max_buffer_size=100)
        # Use explicit casting to help mypy understand the types
        self.receive_stream = cast(
            MemoryObjectReceiveStream[T_Stream | Exception], streams[0]
        )
        self.send_stream = cast(MemoryObjectSendStream[T_Stream], streams[1])


@asynccontextmanager
async def create_test_server_session() -> AsyncGenerator[
    tuple[
        ServerSession,
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage],  # Client read stream
            MemoryObjectSendStream[JSONRPCMessage],  # Client write stream
        ],
    ],
    None,
]:
    """Create a test server session with streams."""
    # Create the streams
    # First stream: Client write -> Server read (with Exception support)
    send_exc, recv_exc = anyio.create_memory_object_stream[JSONRPCMessage | Exception]()

    # Second stream: Server write -> Client read
    send_msg, recv_msg = anyio.create_memory_object_stream[JSONRPCMessage]()

    # Create server session
    session = ServerSession(
        read_stream=recv_exc,  # Server reads (receives) from client's send stream
        write_stream=send_msg,  # Server writes (sends) to client's receive stream
        init_options=InitializationOptions(
            server_name="llmling-server",
            server_version="1.0.0",
            capabilities=ServerCapabilities(
                resources=ResourcesCapability(subscribe=True, listChanged=True),
                prompts=PromptsCapability(listChanged=True),
                tools=ToolsCapability(listChanged=True),
                logging=LoggingCapability(),
            ),
        ),
    )

    async with session:
        # Return the client's ends of the streams
        yield session, (recv_msg, send_exc)
