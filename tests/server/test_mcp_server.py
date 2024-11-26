"""Tests for MCP protocol server implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anyio
from mcp.client.session import ClientSession
from mcp.types import TextContent
import pytest

from llmling.config.models import TextResource


if TYPE_CHECKING:
    from llmling.server import LLMLingServer


@pytest.mark.asyncio
async def test_list_resources(running_server: tuple[LLMLingServer, tuple[Any, Any]]):
    """Test resource listing."""
    server, (client_read, client_write) = running_server

    # Add test resource
    server.config.contexts["test"] = TextResource(
        context_type="text", content="test content", description="Test resource"
    )

    async with ClientSession(client_read, client_write) as session:
        await session.initialize()

        result = await session.list_resources()
        assert len(result.resources) == 1
        assert result.resources[0].name == "test"


@pytest.mark.asyncio
async def test_tool_execution_with_progress(
    running_server: tuple[LLMLingServer, tuple[Any, Any]],
):
    """Test tool execution with progress tracking."""
    server, (client_read, client_write) = running_server

    progress_updates = []

    async def handle_progress(token: str, progress: float, total: float | None) -> None:
        progress_updates.append((progress, total))

    async with ClientSession(client_read, client_write) as session:
        await session.initialize()

        # Set up progress handler
        session.on_progress_notification = handle_progress

        with anyio.move_on_after(2.0):
            # Call tool with progress token
            result = await session.call_tool(
                name="example",
                arguments={
                    "text": "test",
                    "repeat": 2,
                    "_meta": {"progressToken": "test-progress"},
                },
            )

            # Allow time for progress notifications
            await anyio.sleep(0.1)

            # Verify progress updates were received
            assert len(progress_updates) >= 1  # At least one update
            if len(progress_updates) >= 2:  # noqa: PLR2004
                assert progress_updates[0] == (0.0, 1.0)  # Initial progress
                assert progress_updates[-1] == (1.0, 1.0)  # Complete

            # Verify result
            assert not result.isError
            assert len(result.content) == 1
            assert isinstance(result.content[0], TextContent)
            assert result.content[0].text == "testtest"


@pytest.mark.asyncio
async def test_prompt_handling(running_server: tuple[LLMLingServer, tuple[Any, Any]]):
    """Test prompt listing and retrieval."""
    server, (client_read, client_write) = running_server

    # Add test prompt
    server.session.prompt_registry.register(
        "test",
        {
            "name": "test",
            "description": "Test prompt",
            "messages": [{"role": "user", "content": "Test message"}],
        },
    )

    async with ClientSession(client_read, client_write) as session:
        await session.initialize()

        # List prompts
        prompts = await session.list_prompts()
        assert len(prompts.prompts) == 1
        assert prompts.prompts[0].name == "test"

        # Get prompt
        result = await session.get_prompt("test")
        assert len(result.messages) == 1
        assert result.messages[0].content.text == "Test message"
