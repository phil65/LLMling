"""Tests for main server functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anyio
from mcp.client.session import ClientSession
from mcp.types import TextContent
import pytest

from llmling.server.session import SessionState


if TYPE_CHECKING:
    from llmling.server import LLMLingServer


@pytest.mark.asyncio
async def test_server_lifecycle(running_server: tuple[LLMLingServer, tuple[Any, Any]]):
    """Test server startup and shutdown."""
    server, (client_read, client_write) = running_server

    async with ClientSession(client_read, client_write) as session:
        with anyio.move_on_after(2.0):
            # Initialize client
            await session.initialize()

            # Verify server state
            assert server.session.state == SessionState.RUNNING

            # Test basic functionality
            resources = await session.list_resources()
            assert isinstance(resources.resources, list)

            tools = await session.list_tools()
            assert len(tools.tools) >= 1  # At least one test tool


# @pytest.mark.asyncio
# async def test_tool_execution(running_server: tuple[LLMLingServer, tuple[Any, Any]]):
#     """Test tool execution."""
#     server, (client_read, client_write) = running_server

#     async with ClientSession(client_read, client_write) as session:
#         await session.initialize()

#         # Call example tool
#         result = await session.call_tool(
#             name="example",
#             arguments={
#                 "text": "test",
#                 "repeat": 3,
#                 "_meta": {"progressToken": "test-token"},  # Add as part of arguments
#             },
#         )

#         assert not result.isError
#         assert len(result.content) == 1
#         assert result.content[0].text == "testtesttest"


@pytest.mark.asyncio
async def test_error_handling(running_server: tuple[LLMLingServer, tuple[Any, Any]]):
    """Test server error handling."""
    server, (client_read, client_write) = running_server

    async with ClientSession(client_read, client_write) as session:
        await session.initialize()

        # Test invalid tool
        result = await session.call_tool(
            name="nonexistent",
            arguments={},  # Empty arguments for non-existent tool
        )
        assert result.isError
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert "error" in result.content[0].text.lower()
