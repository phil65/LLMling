"""MCP protocol server implementation."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Self

import mcp
from mcp.server import NotificationOptions, Server
from mcp.types import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    Completion,
    CompletionArgument,
    GetPromptResult,
    TextContent,
)
from pydantic import AnyUrl

from llmling.config.models import PathResource, SourceResource
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.server import conversions
from llmling.server.log import configure_server_logging, run_logging_processor
from llmling.server.observers import PromptObserver, ResourceObserver, ToolObserver


if TYPE_CHECKING:
    from collections.abc import Coroutine
    import os

    from llmling.config.runtime import RuntimeConfig
    from llmling.prompts.models import Prompt


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

        # Create MCP server
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
    def from_config_file(
        cls,
        config_path: str | os.PathLike[str],
        *,
        name: str = "llmling-server",
    ) -> LLMLingServer:
        """Create server from config file.

        This is a convenience method that uses the factory internally.
        For more control over server creation, use create_server() directly.

        Args:
            config_path: Path to configuration file
            name: Optional server name

        Returns:
            Configured server instance
        """
        from llmling.server.factory import create_server

        return create_server(config_path, name=name)

    def _create_task(self, coro: Coroutine[None, None, Any]) -> asyncio.Task[Any]:
        """Create and track an asyncio task."""
        task: asyncio.Task[Any] = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _setup_handlers(self) -> None:
        """Register MCP protocol handlers."""

        @self.server.set_logging_level()
        async def handle_set_level(level: mcp.LoggingLevel) -> None:
            """Handle logging level changes."""
            level_map = {
                "debug": logging.DEBUG,
                "info": logging.INFO,
                "notice": logging.INFO,
                "warning": logging.WARNING,
                "error": logging.ERROR,
                "critical": logging.CRITICAL,
                "alert": logging.CRITICAL,
                "emergency": logging.CRITICAL,
            }
            try:
                python_level = level_map[level]
                logger.setLevel(python_level)
                await self.current_session.send_log_message(
                    level="info",
                    data=f"Log level set to {level}",
                    logger=self.name,
                )
            except Exception as exc:
                error = mcp.McpError("Error setting log level")
                error.error = mcp.ErrorData(code=INTERNAL_ERROR, message=str(exc))
                raise error from exc

        @self.server.list_tools()
        async def handle_list_tools() -> list[mcp.types.Tool]:
            """Handle tools/list request."""
            return [conversions.to_mcp_tool(tool) for tool in self.runtime.get_tools()]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str,
            arguments: dict[str, Any] | None = None,
        ) -> list[TextContent]:
            """Handle tools/call request."""
            try:
                # Filter out _meta from arguments
                tool_args = {
                    k: v for k, v in (arguments or {}).items() if not k.startswith("_")
                }

                result = await self.runtime.execute_tool(name, **tool_args)
                return [TextContent(type="text", text=str(result))]
            except Exception as exc:
                logger.exception("Tool execution failed: %s", name)
                error_msg = f"Tool execution failed: {exc}"
                return [TextContent(type="text", text=error_msg)]

        @self.server.list_prompts()
        async def handle_list_prompts() -> list[mcp.types.Prompt]:
            """Handle prompts/list request."""
            return [
                conversions.to_mcp_prompt(prompt) for prompt in self.runtime.get_prompts()
            ]

        @self.server.get_prompt()
        async def handle_get_prompt(
            name: str,
            arguments: dict[str, str] | None = None,
        ) -> GetPromptResult:
            """Handle prompts/get request."""
            try:
                prompt = self.runtime.get_prompt(name)
                messages = await self.runtime.render_prompt(name, arguments)

                return GetPromptResult(
                    description=prompt.description,
                    messages=[conversions.to_mcp_message(msg) for msg in messages],
                )
            except exceptions.LLMLingError as exc:
                msg = str(exc)
                error = mcp.McpError(msg)
                error.error = mcp.ErrorData(
                    code=INVALID_PARAMS if "not found" in msg else INTERNAL_ERROR,
                    message=msg,
                )
                raise error from exc

        @self.server.list_resources()
        async def handle_list_resources() -> list[mcp.types.Resource]:
            """Handle resources/list request."""
            resources = []
            for name in self.runtime.list_resources():
                try:
                    # First get URI and basic info without loading
                    uri = self.runtime.get_resource_uri(name)
                    resource_config = self.runtime._config.resources[
                        name
                    ]  # Get raw config

                    mcp_resource = mcp.types.Resource(
                        uri=conversions.to_mcp_uri(uri),
                        name=name,
                        description=resource_config.description,
                        mimeType="text/plain",  # Default, could be made more specific
                    )
                    resources.append(mcp_resource)

                except Exception:
                    logger.exception(
                        "Failed to create resource listing for %r. Config: %r",
                        name,
                        self.runtime._config.resources.get(name),
                    )
                    continue

            return resources

        @self.server.read_resource()
        async def handle_read_resource(uri: mcp.types.AnyUrl) -> str | bytes:
            """Handle direct resource content requests."""
            try:
                internal_uri = conversions.from_mcp_uri(str(uri))
                logger.debug("Loading resource from internal URI: %s", internal_uri)

                if "://" not in internal_uri:
                    resource = await self.runtime.load_resource(internal_uri)
                else:
                    resource = await self.runtime.load_resource_by_uri(internal_uri)

                if resource.metadata.mime_type.startswith("text/"):
                    return resource.content
                return resource.content_items[0].content.encode()

            except Exception as exc:
                msg = f"Failed to read resource: {exc}"
                logger.exception(msg)
                error = mcp.McpError(msg)
                error.error = mcp.ErrorData(code=INTERNAL_ERROR, message=str(exc))
                raise error from exc

        @self.server.completion()
        async def handle_completion(
            ref: mcp.types.PromptReference | mcp.types.ResourceReference,
            argument: CompletionArgument,
        ) -> Completion | None:
            """Handle completion requests."""
            try:
                match ref:
                    case mcp.types.PromptReference():
                        return await self._complete_prompt_argument(
                            self.runtime.get_prompt(ref.name),
                            argument.name,
                            argument.value,
                        )
                    case mcp.types.ResourceReference():
                        url = AnyUrl(ref.uri)
                        return await self._complete_resource(url, argument)
                    case _:
                        msg = f"Invalid reference type: {type(ref)}"
                        error = mcp.McpError(msg)
                        error.error = mcp.ErrorData(code=INVALID_PARAMS, message=msg)
                        raise error  # noqa: TRY301
            except Exception:
                logger.exception("Completion failed")
                return None  # Let MCP convert to empty Completion

        @self.server.progress_notification()
        async def handle_progress(
            token: str | int,
            progress: float,
            total: float | None,
        ) -> None:
            """Handle progress notifications from client."""
            logger.debug(
                "Progress notification: %s %.1f/%.1f",
                token,
                progress,
                total or 0.0,
            )

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
            # Initialize runtime
            await self.runtime.startup()

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
        """Get current session from request context."""
        try:
            return self.server.request_context.session
        except LookupError as exc:
            msg = "No active request context"
            raise RuntimeError(msg) from exc

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
            self._create_task(
                session.send_progress_notification(
                    progress_token=token,
                    progress=progress,
                    total=total,
                )
            )

            # Optionally send description as log message
            if description:
                self._create_task(
                    session.send_log_message(
                        level="info",
                        data=description,
                    )
                )

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
        """Notify clients about resource changes."""
        try:
            url = conversions.to_mcp_uri(uri)
            self._create_task(self.current_session.send_resource_updated(url))
            self._create_task(self.current_session.send_resource_list_changed())
        except RuntimeError:
            logger.debug("No active session for notification")
        except Exception:
            logger.exception("Failed to send resource change notification")

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

    async def _complete_prompt_argument(
        self,
        prompt: Prompt,
        arg_name: str,
        current_value: str,
    ) -> Completion:
        """Generate completions for prompt arguments."""
        try:
            # Get completions through runtime
            completions = await self.runtime.get_prompt_completions(
                prompt.name, arg_name, current_value
            )

            # Add any available defaults if no current value
            arg = next((a for a in prompt.arguments if a.name == arg_name), None)
            if arg and not current_value:
                if arg.default is not None:
                    completions.append(str(arg.default))

                # Add description-based suggestions
                if arg.description and "one of:" in arg.description:
                    try:
                        options_part = arg.description.split("one of:", 1)[1]
                        options = [opt.strip() for opt in options_part.split(",")]
                        completions.extend(
                            opt for opt in options if opt.startswith(current_value)
                        )
                    except IndexError:
                        pass

            # Deduplicate while preserving order
            seen = set()
            unique_completions = [
                x
                for x in completions
                if not (x in seen or seen.add(x))  # type: ignore
            ]

            return Completion(
                values=unique_completions[:100],
                total=len(unique_completions),
                hasMore=len(unique_completions) > 100,  # noqa: PLR2004
            )

        except Exception:
            logger.exception(
                "Completion failed for prompt=%s argument=%s", prompt.name, arg_name
            )
            return Completion(values=[], total=0, hasMore=False)

    async def _complete_resource(
        self,
        uri: AnyUrl,
        argument: CompletionArgument,
    ) -> Completion:
        """Generate completions for resource fields."""
        try:
            # Convert AnyUrl to string for resource lookup
            str_uri = str(uri)
            resource = await self.runtime.load_resource_by_uri(str_uri)
            values: list[str] = []

            # Different completion logic based on resource type
            match resource:
                case PathResource():
                    # Complete paths
                    import glob

                    pattern = f"{argument.value}*"
                    values = list(glob.glob(pattern))  # noqa: PTH207

                case SourceResource():
                    # Complete Python import paths
                    values = [
                        name
                        for name in self._get_importable_names()
                        if name.startswith(argument.value)
                    ]

            return Completion(
                values=values[:100],
                total=len(values),
                hasMore=len(values) > 100,  # noqa: PLR2004
            )

        except Exception:
            logger.exception("Resource completion failed")
            return Completion(values=[], total=0, hasMore=False)


if __name__ == "__main__":
    import asyncio
    import sys

    from llmling import config_resources

    config_path = sys.argv[1] if len(sys.argv) > 1 else config_resources.TEST_CONFIG
    # asyncio.run(serve(config_path))
