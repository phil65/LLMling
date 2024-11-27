"""Tests for MCP protocol implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from llmling.config.models import Config, TextResource, ToolConfig
from llmling.prompts.models import Prompt, PromptMessage
from llmling.testing.testclient import HandshakeClient


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


@pytest.fixture
def test_config() -> Config:
    """Create test configuration."""
    prompt = Prompt(
        name="test",
        description="test",
        messages=[PromptMessage(role="system", content="test")],
    )
    resource = TextResource(
        resource_type="text",
        content="Test content",
        description="Test resource",
    )
    tool_cfg = ToolConfig(
        import_path="llmling.testing.tools.example_tool",
        name="example",
        description="Test tool",
    )
    return Config(
        version="1.0",
        prompts={"test": prompt},
        resources={"test": resource},
        tools={"example": tool_cfg},
    )


@pytest.fixture
async def config_file(tmp_path: Path, test_config: Config) -> Path:
    """Create temporary config file."""
    config_path = tmp_path / "test_config.yml"
    content = test_config.model_dump(exclude_none=True)
    with config_path.open("w") as f:
        yaml.dump(content, f)
    return config_path


@pytest.fixture
async def configured_client(config_file: Path) -> AsyncGenerator[HandshakeClient, None]:
    """Create client with test configuration."""
    client = HandshakeClient(config_path=str(config_file))
    try:
        await client.start()
        response = await client.do_handshake()
        assert response["serverInfo"]["name"] == "llmling-server"
        yield client
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_resource_operations(configured_client: HandshakeClient):
    """Test MCP resource operations."""
    response = await configured_client.send_request("resources/list")
    assert "resources" in response
    resource_list = response["resources"]
    assert len(resource_list) >= 1
    assert resource_list[0]["name"] == "Test resource"


@pytest.mark.asyncio
async def test_mcp_tool_operations(configured_client: HandshakeClient):
    """Test MCP tool operations."""
    # First verify tool exists
    tools = await configured_client.send_request("tools/list")
    tools_list = tools["tools"]
    assert len(tools_list) >= 1
    assert tools_list[0]["name"] == "example"

    # Now call it
    response = await configured_client.send_request(
        "tools/call",
        {
            "name": "example",
            "arguments": {"text": "test", "repeat": 1},
        },
    )
    assert "content" in response
    assert len(response["content"]) == 1
    assert response["content"][0]["text"] == "test"


@pytest.mark.asyncio
async def test_mcp_error_handling(configured_client: HandshakeClient):
    """Test MCP error response format."""
    response = await configured_client.send_request("tools/call", {"name": "nonexistent"})
    assert "content" in response
    assert len(response["content"]) == 1
    assert "not found" in response["content"][0]["text"].lower()


@pytest.mark.asyncio
async def test_mcp_handshake(configured_client: HandshakeClient):
    """Test MCP protocol handshake."""
    # Do another handshake to explicitly test it
    init_response = await configured_client.do_handshake()

    # Verify server info
    assert "serverInfo" in init_response
    assert init_response["serverInfo"]["name"] == "llmling-server"

    # Verify capabilities (they're on the root level, not in serverInfo)
    assert "capabilities" in init_response
    capabilities = init_response["capabilities"]

    # Verify specific capabilities
    assert "resources" in capabilities
    assert "prompts" in capabilities
    assert "tools" in capabilities


@pytest.mark.asyncio
async def test_mcp_prompt_operations(configured_client: HandshakeClient):
    """Test MCP prompt operations."""
    # List prompts
    prompts = await configured_client.send_request("prompts/list")
    assert "prompts" in prompts
    assert isinstance(prompts["prompts"], list)

    # Get specific prompt
    # Add a test prompt if needed
    if prompts["prompts"]:
        test_prompt = prompts["prompts"][0]
        result = await configured_client.send_request(
            "prompts/get",
            {"name": test_prompt["name"], "arguments": None},
        )
        assert "messages" in result
        assert isinstance(result["messages"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
