# llmling/testing/testclient.py
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
from typing import Any

from llmling.core.log import get_logger


logger = get_logger(__name__)


class HandshakeClient:
    """Test client for MCP protocol."""

    def __init__(self, server_command: list[str]) -> None:
        self.server_command = server_command
        self.process: subprocess.Popen[bytes] | None = None
        self._stderr_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start server process and do handshake."""
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self.process = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
        )

        # Start stderr reader
        async def read_stderr():
            while self.process and self.process.stderr:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, self.process.stderr.readline
                )
                if not line:
                    break
                logger.debug("Server stderr: %s", line.decode().strip())

        self._stderr_task = asyncio.create_task(read_stderr())
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
                response = json.loads(line.decode())
                logger.debug("Received: %s", response)
            except json.JSONDecodeError:
                # Skip non-JSON lines
                logger.debug("Server output: %s", line.decode().strip())
                continue
            else:
                return response

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None, timeout: float = 10.0
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

        request_str = json.dumps(request) + "\n"
        logger.debug("Sending request: %s", request_str.strip())
        self.process.stdin.write(request_str.encode())
        self.process.stdin.flush()

        try:
            async with asyncio.timeout(timeout):
                while True:
                    response = await self._read_response()
                    if "id" in response and response["id"] == request["id"]:
                        if "error" in response:
                            msg = f"Server error: {response['error']}"
                            raise RuntimeError(msg)
                        return response.get("result")
        except TimeoutError:
            logger.exception("Timeout waiting for response to: %s", method)
            raise

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
        logger.debug("Sending notification: %s", notification_str.strip())
        self.process.stdin.write(notification_str.encode())
        self.process.stdin.flush()

    async def do_handshake(self) -> dict[str, Any]:
        """Perform initial handshake with server."""
        init_response = await self.send_request(
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
        await self.send_notification("notifications/initialized", {})
        return init_response

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools."""
        result = await self.send_request("tools/list")
        return result["tools"]

    async def list_resources(self) -> list[dict[str, Any]]:
        """List available resources."""
        result = await self.send_request("resources/list")
        return result["resources"]

    async def list_prompts(self) -> list[dict[str, Any]]:
        """List available prompts."""
        result = await self.send_request("prompts/list")
        return result["prompts"]

    async def call_tool(
        self, name: str, arguments: dict[str, Any], with_progress: bool = False
    ) -> dict[str, Any]:
        """Call a tool."""
        if with_progress:
            arguments["_meta"] = {"progressToken": f"progress-{name}"}
        return await self.send_request(
            "tools/call", {"name": name, "arguments": arguments}
        )

    async def close(self) -> None:
        """Stop server process."""
        if self._stderr_task:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task

        if self.process:
            try:
                await self.send_request("shutdown", {}, timeout=2.0)
                await self.send_notification("notifications/exit", {})
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning("Error during shutdown: %s", e)
            finally:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
