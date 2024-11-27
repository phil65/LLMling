"""Test server lifecycle using different approaches."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import TYPE_CHECKING, Any

import logfire
import pytest


if TYPE_CHECKING:
    from llmling.server import LLMLingServer
    from llmling.testing.testclient import HandshakeClient


# Initialize logfire to avoid warnings
logfire.configure()


@pytest.mark.asyncio
async def test_server_lifecycle_handshake_client(client: HandshakeClient) -> None:
    """Test server lifecycle using HandshakeClient."""
    try:
        await client.start()
        await asyncio.sleep(0.5)

        # Initialize connection
        init_response = await client.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        )
        assert isinstance(init_response, dict)
        assert "serverInfo" in init_response
        assert init_response["serverInfo"]["name"] == "llmling-server"

        # Send initialized notification
        await client.send_notification("notifications/initialized", {})

        # Test tool listing
        tools_response = await client.send_request("tools/list")
        assert isinstance(tools_response, dict)
        assert "tools" in tools_response
        assert isinstance(tools_response["tools"], list)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_server_lifecycle_test_session(
    running_server: tuple[LLMLingServer, tuple[Any, Any]],
) -> None:
    """Test server lifecycle using test session."""
    server, (client_read, client_write) = running_server

    # Create proper MCP message
    from mcp.types import JSONRPCMessage, JSONRPCRequest

    init_request = JSONRPCMessage(
        JSONRPCRequest(
            jsonrpc="2.0",
            id=1,
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        )
    )

    await client_write.send(init_request)
    response = await client_read.receive()
    assert "result" in response.root.model_dump()
    assert "serverInfo" in response.root.result

    # Send initialized notification
    from mcp.types import JSONRPCNotification

    notification = JSONRPCMessage(
        JSONRPCNotification(
            jsonrpc="2.0",
            method="notifications/initialized",
            params={},
        )
    )
    await client_write.send(notification)

    # Test tools list
    tools_request = JSONRPCMessage(
        JSONRPCRequest(
            jsonrpc="2.0",
            id=2,
            method="tools/list",
            params={},
        )
    )
    await client_write.send(tools_request)
    tools_response = await client_read.receive()
    assert "tools" in tools_response.root.result


@pytest.mark.asyncio
async def test_server_lifecycle_direct(server: LLMLingServer) -> None:
    """Test server lifecycle using direct method calls."""
    try:
        # Start registries
        await server.processor_registry.startup()
        await server.tool_registry.startup()

        # Test registry contents directly
        tools = list(server.tool_registry.values())
        assert isinstance(tools, list)
        assert len(tools) > 0  # Should have our test tools
    finally:
        await server.shutdown()


@pytest.mark.asyncio
async def test_server_lifecycle_subprocess() -> None:
    """Test server lifecycle using raw subprocess."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "llmling.server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Start stderr reader task
    async def read_stderr():
        assert process.stderr
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            print(f"Server stderr: {line.decode().strip()}")

    stderr_task = asyncio.create_task(read_stderr())

    try:
        assert process.stdin and process.stdout
        await asyncio.sleep(0.5)  # Give server time to start

        # Send initialize request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        }
        process.stdin.write(json.dumps(request).encode() + b"\n")
        await process.stdin.drain()

        # Read and verify response
        response = await process.stdout.readline()
        if not response:
            raise RuntimeError("No response from server")
        result = json.loads(response.decode())
        assert "result" in result
        assert "serverInfo" in result["result"]

        # Send initialized notification
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
        process.stdin.write(json.dumps(notification).encode() + b"\n")
        await process.stdin.drain()

        # Send tools list request
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        }
        process.stdin.write(json.dumps(tools_request).encode() + b"\n")
        await process.stdin.drain()

        # Read and verify tools response
        tools_response = await process.stdout.readline()
        if not tools_response:
            msg = "No tools response from server"
            raise RuntimeError(msg)
        tools_result = json.loads(tools_response.decode())
        assert "result" in tools_result
        assert "tools" in tools_result["result"]
        assert isinstance(tools_result["result"]["tools"], list)

    finally:
        stderr_task.cancel()
        process.terminate()
        await process.wait()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
