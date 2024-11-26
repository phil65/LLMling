"""Tests for MCP protocol server implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from llmling.testing.testclient import HandshakeClient


@pytest.mark.asyncio
async def test_list_resources(client: HandshakeClient):
    """Test resource listing."""
    await client.start()
    await client.do_handshake()

    # MCP wraps response in a ListResourcesResult
    response = await client.send_request("resources/list")
    assert "resources" in response
    assert isinstance(response["resources"], list)


@pytest.mark.asyncio
async def test_tool_execution_with_progress(client: HandshakeClient):
    """Test tool execution with progress tracking."""
    await client.start()
    await client.do_handshake()

    # MCP wraps tool response in CallToolResult
    response = await client.send_request(
        "tools/call",
        {
            "name": "example",
            "arguments": {
                "text": "test",
                "repeat": 2,
                "_meta": {"progressToken": "test-progress"},
            },
        },
    )
    assert not response["isError"]
    assert "content" in response
    assert len(response["content"]) == 1
    assert response["content"][0]["text"] == "testtest"


@pytest.mark.asyncio
async def test_prompt_handling(client: HandshakeClient):
    """Test prompt listing and retrieval."""
    await client.start()
    await client.do_handshake()

    # List prompts - returns ListPromptsResult
    prompts = await client.send_request("prompts/list")
    assert "prompts" in prompts
    assert isinstance(prompts["prompts"], list)

    # Get prompt - returns GetPromptResult
    result = await client.send_request(
        "prompts/get",
        {"name": "test-prompt", "arguments": None},
    )
    assert "messages" in result
    assert isinstance(result["messages"], list)


@pytest.mark.asyncio
async def test_error_handling(client: HandshakeClient):
    """Test server error handling."""
    await client.start()
    await client.do_handshake()

    # Test calling non-existent tool
    with pytest.raises(RuntimeError) as exc_info:
        await client.send_request(
            "tools/call",
            {"name": "nonexistent", "arguments": {}},
        )

    error_msg = str(exc_info.value)
    assert "not found" in error_msg.lower()


@pytest.mark.asyncio
async def test_server_lifecycle(client: HandshakeClient):
    """Test server startup and shutdown."""
    try:
        await client.start()
        init_response = await client.do_handshake()

        assert "serverInfo" in init_response
        assert init_response["serverInfo"]["name"] == "llmling-server"

        # Test basic functionality
        tools = await client.send_request("tools/list")
        assert "tools" in tools

        resources = await client.send_request("resources/list")
        assert "resources" in resources

    finally:
        await client.close()
