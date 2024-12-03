"""MCP protocol server implementation."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Self

import mcp
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl

from llmling.config.manager import ConfigManager
from llmling.config.runtime import RuntimeConfig
from llmling.core.log import get_logger
from llmling.server.handlers import register_handlers
from llmling.server.log import configure_server_logging, run_logging_processor
from llmling.server.observers import PromptObserver, ResourceObserver, ToolObserver


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Coroutine
    import os


logger = get_logger(__name__)


class LLMLingServer:
    """MCP protocol server implementation."""

    def __init__(
        self,
        runtime: RuntimeConfig,
        *,
        name: str = "llmling-server",
    ) -> None:
        """Initialize server with runtime configuration.

        Args:
            runtime: Fully initialized runtime configuration
            name: Server name for MCP protocol
        """
        self.name = name
        self.runtime = runtime
        self._subscriptions: defaultdict[str, set[mcp.ServerSession]] = defaultdict(set)
        self.server = Server(name)
        self.server.notification_options = NotificationOptions(
            prompts_changed=True,
            resources_changed=True,
            tools_changed=True,
        )
        self._tasks: set[asyncio.Task[Any]] = set()

        self._setup_handlers()
        self._setup_observers()

    @classmethod
    @asynccontextmanager
    async def from_config_file(
        cls,
        config_path: str | os.PathLike[str],
        *,
        name: str = "llmling-server",
    ) -> AsyncIterator[LLMLingServer]:
        """Create and run server from config file with proper context management.

        Args:
            config_path: Path to configuration file
            name: Optional server name

        Example:
            ```python
            async with LLMLingServer.from_config_file("config.yml") as server:
                await server.start()
            ```
        """
        manager = ConfigManager.load(config_path)
        async with RuntimeConfig.from_config(manager.config) as runtime:
            server = cls(runtime, name=name)
            try:
                yield server
            finally:
                await server.shutdown()

    def _create_task(self, coro: Coroutine[None, None, Any]) -> asyncio.Task[Any]:
        """Create and track an asyncio task."""
        task: asyncio.Task[Any] = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _setup_handlers(self) -> None:
        """Register MCP protocol handlers."""
        register_handlers(self)

    def _setup_observers(self) -> None:
        """Set up registry observers for MCP notifications."""
        self.resource_observer = ResourceObserver(self)
        self.prompt_observer = PromptObserver(self)
        self.tool_observer = ToolObserver(self)
        self.runtime.add_resource_observer(self.resource_observer.events)
        self.runtime.add_prompt_observer(self.prompt_observer.events)
        self.runtime.add_tool_observer(self.tool_observer.events)

    async def start(self, *, raise_exceptions: bool = False) -> None:
        """Start the server."""
        try:
            # Start MCP server
            handler = configure_server_logging(self.server)
            options = self.server.create_initialization_options()
            async with (
                mcp.stdio_server() as (read_stream, write_stream),
                asyncio.TaskGroup() as tg,
            ):
                tg.create_task(run_logging_processor(handler))
                await self.server.run(
                    read_stream,
                    write_stream,
                    options,
                    raise_exceptions=raise_exceptions,
                )
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Shutdown the server."""
        try:
            # Cancel all pending tasks
            if self._tasks:
                for task in self._tasks:
                    task.cancel()
                await asyncio.gather(*self._tasks, return_exceptions=True)
                self._tasks.clear()

            # Remove observers
            self.runtime.remove_resource_observer(self.resource_observer.events)
            self.runtime.remove_prompt_observer(self.prompt_observer.events)
            self.runtime.remove_tool_observer(self.tool_observer.events)

            # Shutdown runtime
            await self.runtime.shutdown()
        finally:
            self._tasks.clear()

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Async context manager exit."""
        await self.shutdown()

    @property
    def current_session(self) -> mcp.ServerSession:
        """Get client info from request context."""
        try:
            return self.server.request_context.session
        except LookupError as exc:
            msg = "No active request context"
            raise RuntimeError(msg) from exc

    @property
    def client_info(self) -> mcp.Implementation | None:
        """Get current session from request context."""
        session = self.current_session
        if not session.client_params:
            return None
        return session.client_params.clientInfo

    def notify_progress(
        self,
        token: str,
        progress: float,
        total: float | None = None,
        description: str | None = None,
    ) -> None:
        """Send progress notification to client."""
        try:
            # Get current session
            session = self.current_session

            # Create and track the progress notification task
            task = session.send_progress_notification(
                progress_token=token,
                progress=progress,
                total=total,
            )
            self._create_task(task)

            # Optionally send description as log message
            if description:
                coro = session.send_log_message(level="info", data=description)
                self._create_task(coro)

        except Exception:
            logger.exception("Failed to send progress notification")

    async def notify_resource_list_changed(self) -> None:
        """Notify clients about resource list changes."""
        try:
            await self.current_session.send_resource_list_changed()
        except RuntimeError:
            logger.debug("No active session for notification")
        except Exception:
            logger.exception("Failed to send resource list change notification")

    async def notify_resource_change(self, uri: str) -> None:
        """Notify subscribers about resource changes."""
        if uri in self._subscriptions:
            try:
                await self.current_session.send_resource_updated(AnyUrl(uri))
            except Exception:
                msg = "Failed to notify subscribers about resource change: %s"
                logger.exception(msg, uri)

    async def notify_prompt_list_changed(self) -> None:
        """Notify clients about prompt list changes."""
        try:
            self._create_task(self.current_session.send_prompt_list_changed())
        except RuntimeError:
            logger.debug("No active session for notification")
        except Exception:
            logger.exception("Failed to send prompt list change notification")

    async def notify_tool_list_changed(self) -> None:
        """Notify clients about tool list changes."""
        try:
            self._create_task(self.current_session.send_tool_list_changed())
        except RuntimeError:
            logger.debug("No active session for notification")
        except Exception:
            logger.exception("Failed to send tool list change notification")


if __name__ == "__main__":
    import sys

    from llmling import config_resources

    config_path = sys.argv[1] if len(sys.argv) > 1 else config_resources.TEST_CONFIG
