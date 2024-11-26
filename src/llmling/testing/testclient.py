from __future__ import annotations

import asyncio
import json
import os
import subprocess
from typing import Any


class HandshakeClient:
    """Minimal client to test MCP handshake."""

    def __init__(self, server_command: list[str]) -> None:
        self.server_command = server_command
        self.process: subprocess.Popen[bytes] | None = None

    async def start(self) -> None:
        """Start server process."""
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self.process = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
        )
        # Give server time to start
        await asyncio.sleep(0.5)

    async def _read_response(self) -> dict[str, Any]:
        """Read JSON-RPC response from server."""
        if not self.process or not self.process.stdout:
            msg = "Server process not available"
            raise RuntimeError(msg)

        while True:
            line = await asyncio.get_event_loop().run_in_executor(
                None, self.process.stdout.readline
            )
            if not line:
                msg = "Server closed connection"
                raise RuntimeError(msg)

            try:
                return json.loads(line.decode())
            except json.JSONDecodeError:
                # Skip non-JSON lines
                continue

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Send JSON-RPC request and get response."""
        if not self.process or not self.process.stdin:
            msg = "Server not started"
            raise RuntimeError(msg)

        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1,
        }

        # Send request
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str.encode())
        self.process.stdin.flush()

        # Wait for response with timeout
        async with asyncio.timeout(5.0):
            while True:
                response = await self._read_response()
                if "id" in response and response["id"] == request["id"]:
                    if "error" in response:
                        msg = f"Server error: {response['error']}"
                        raise RuntimeError(msg)
                    return response.get("result")

    async def send_notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        """Send JSON-RPC notification."""
        if not self.process or not self.process.stdin:
            msg = "Server not started"
            raise RuntimeError(msg)

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }

        notification_str = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_str.encode())
        self.process.stdin.flush()

    async def close(self) -> None:
        """Stop server process."""
        if self.process:
            try:
                # Proper shutdown sequence
                await self.send_request("shutdown", {})
                await self.send_notification("notifications/exit", {})
                await asyncio.sleep(0.1)
            except Exception:
                pass
            finally:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
