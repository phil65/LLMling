from __future__ import annotations

import asyncio
import sys

import pytest

from llmling.testing.testclient import HandshakeClient


@pytest.mark.asyncio
async def test_server_handshake() -> None:
    """Test basic MCP handshake sequence."""
    client = HandshakeClient([sys.executable, "-m", "llmling.server"])
    try:
        # Start server
        await client.start()

        # Send initialize request
        init_response = await client.send_request(
            "initialize",
            {
                "protocolVersion": "0.1",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
                "processId": None,
                "rootUri": None,
                "workspaceFolders": None,
            },
        )

        # Check initialization response
        assert isinstance(init_response, dict)
        assert "serverInfo" in init_response
        assert init_response["serverInfo"]["name"] == "llmling-server"
        assert "capabilities" in init_response
        assert "tools" in init_response["capabilities"]

        # Send initialized notification
        await client.send_notification("notifications/initialized", {})

        # List tools (should work but might be empty)
        tools_response = await client.send_request("tools/list")
        assert isinstance(tools_response, dict)
        assert "tools" in tools_response
        assert isinstance(tools_response["tools"], list)

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_server_handshake())
