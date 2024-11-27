"""Tests for MCP protocol implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from llmling.config.models import Config, TextResource, ToolConfig
from llmling.testing.testclient import HandshakeClient


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


@pytest.fixture
def test_config() -> Config:
    """Create test configuration."""
    return Config(
        version="1.0",
        contexts={
            "test": TextResource(
                context_type="text",  # Required field
                content="Test content",
                description="Test resource",
            )
        },
        tools={
            "example": ToolConfig(
                import_path="llmling.testing.tools.example_tool",
                name="example",
                description="Test tool",
            )
        },
    )


@pytest.fixture
async def config_file(tmp_path: Path, test_config: Config) -> Path:
    """Create temporary config file."""
    config_path = tmp_path / "test_config.yml"
    # Use yaml.dump directly to ensure proper YAML formatting
    content = test_config.model_dump(exclude_none=True)
    with config_path.open("w") as f:
        yaml.dump(content, f)

    # Debug output
    print(f"\nCreated config file at: {config_path}")
    print(f"Config content:\n{config_path.read_text()}")
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
