from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
from typing import Any

from llmling.core.log import get_logger


logger = get_logger(__name__)


class TestClient:
    def __init__(self, server_command: list[str]) -> None:
        self.server_command = server_command
        self.process: subprocess.Popen[bytes] | None = None

    async def start(self) -> None:
        """Start the server process."""
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self.process = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
        )
        logger.debug("Started server process")
        await asyncio.sleep(0.5)

    async def _read_response(self) -> dict[str, Any]:
        """Read a response from the server."""
        if not self.process or not self.process.stdout:
            msg = "Server process not available"
            raise RuntimeError(msg)

        # Also read stderr in background to see server logs
        async def read_stderr():
            while True:
                if not self.process or not self.process.stderr:
                    break
                err = await asyncio.get_event_loop().run_in_executor(
                    None, self.process.stderr.readline
                )
                if not err:
                    break
                logger.debug("Server stderr: %s", err.decode().strip())

        # Start stderr reader and store the task
        self._stderr_task = asyncio.create_task(read_stderr())

        # Read stdout
        line = await asyncio.get_event_loop().run_in_executor(
            None, self.process.stdout.readline
        )

        if not line:
            msg = "Server closed connection"
            raise RuntimeError(msg)

        try:
            response = json.loads(line.decode())
            logger.debug("Received JSON: %s", response)
        except json.JSONDecodeError:
            # Log non-JSON lines as debug messages
            logger.debug("Server stdout: %s", line.decode().strip())
            # Keep reading until we get JSON
            return await self._read_response()
        else:
            return response

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Send JSON-RPC request to server."""
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
        logger.debug("Sending: %s", request_str.strip())
        self.process.stdin.write(request_str.encode())
        self.process.stdin.flush()

        # Wait for response with timeout
        try:
            async with asyncio.timeout(5.0):
                while True:
                    response = await self._read_response()
                    if "id" in response and response["id"] == request["id"]:
                        if "error" in response:
                            msg = f"Server error: {response['error']}"
                            raise RuntimeError(msg)
                        return response.get("result")
        except TimeoutError:
            msg = "Timeout waiting for server response"
            logger.exception(msg)
            raise

    async def send_notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        """Send notification (no response expected)."""
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

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools."""
        logger.debug("Starting tools listing sequence")

        # First initialize
        logger.debug("Sending initialize request")
        response = await self.send_request(
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
        logger.debug("Initialize response: %s", response)

        # Send initialized notification
        logger.debug("Sending initialized notification")
        await self.send_notification("notifications/initialized", {})

        # Now list tools
        logger.debug("Sending tools/list request")
        return await self.send_request("tools/list")

    async def close(self) -> None:
        """Stop the server."""
        if hasattr(self, "_stderr_task"):
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task

        if self.process:
            try:
                await self.send_request("shutdown", {})
                await self.send_notification("exit", {})
            except Exception as e:  # noqa: BLE001
                logger.warning("Error during shutdown: %s", e)
            finally:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()


async def main() -> None:
    import logging

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    client = TestClient([sys.executable, "-m", "llmling.server"])
    try:
        await client.start()
        tools = await client.list_tools()
        print("\nAvailable tools:", tools)
    except Exception:
        logger.exception("Error")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
