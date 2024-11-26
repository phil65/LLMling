"""Session management for LLMling server."""

from __future__ import annotations

import asyncio
from asyncio import Queue
from enum import Enum
from typing import TYPE_CHECKING, Any

import logfire

from llmling.core.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling.config.models import Config
    from llmling.processors.registry import ProcessorRegistry
    from llmling.prompts.registry import PromptRegistry
    from llmling.resources import ResourceLoaderRegistry
    from llmling.tools.registry import ToolRegistry


logger = get_logger(__name__)


class SessionState(Enum):
    """States a session can be in."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    CLOSING = "closing"
    CLOSED = "closed"


class MessageProcessor:
    """Context manager for message processing."""

    def __init__(self, session: LLMLingSession) -> None:
        """Initialize with session."""
        self.session = session
        self._generator = self._process_messages()

    async def _process_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Process messages from queue."""
        if self.session.state != SessionState.RUNNING:
            error_msg = f"Cannot process messages in state: {self.session.state}"
            raise RuntimeError(error_msg)

        with logfire.span("Processing messages"):
            while True:
                try:
                    # Changed variable name to avoid redefinition
                    message: dict[str, Any] = await self.session._message_queue.get()
                    if await self.session._validate_message(message):
                        yield message
                    self.session._message_queue.task_done()
                except asyncio.CancelledError:
                    break
                except Exception:
                    logger.exception("Error processing message")
                    continue

    async def __aenter__(self) -> AsyncIterator[dict[str, Any]]:
        """Enter the context."""
        return self._generator

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        # Clean up any remaining messages
        while not self.session._message_queue.empty():
            try:
                _ = await self.session._message_queue.get()
                self.session._message_queue.task_done()
            except Exception:  # noqa: BLE001
                pass


class LLMLingSession:
    """Manages state and registries for a server session."""

    def __init__(
        self,
        config: Config,
        resource_registry: ResourceLoaderRegistry,
        processor_registry: ProcessorRegistry,
        prompt_registry: PromptRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """Initialize session with registries.

        Args:
            config: Server configuration
            resource_registry: Registry for resource loaders
            processor_registry: Registry for content processors
            prompt_registry: Registry for prompt templates
            tool_registry: Registry for LLM-callable tools
        """
        self.config = config
        self.resource_registry = resource_registry
        self.processor_registry = processor_registry
        self.prompt_registry = prompt_registry
        self.tool_registry = tool_registry
        self._message_queue: Queue[dict[str, Any]] = Queue()
        self._state = SessionState.INITIALIZING
        self._started = False

    @property
    def state(self) -> SessionState:
        """Current session state."""
        return self._state

    @state.setter
    def state(self, new_state: SessionState) -> None:
        """Update session state with validation."""
        valid_transitions = {
            SessionState.INITIALIZING: {SessionState.RUNNING},
            SessionState.RUNNING: {SessionState.CLOSING},
            SessionState.CLOSING: {SessionState.CLOSED},
        }

        if self._state not in valid_transitions or new_state not in valid_transitions.get(
            self._state, set()
        ):
            msg = f"Invalid state transition: {self._state} -> {new_state}"
            raise ValueError(msg)

        self._state = new_state
        logger.info("Session state changed to: %s", new_state)

    def process_messages(self) -> MessageProcessor:
        """Get message processor context manager."""
        return MessageProcessor(self)

    async def send_message(self, message: dict[str, Any]) -> None:
        """Send message to queue.

        Args:
            message: Message to enqueue

        Raises:
            RuntimeError: If session is not running
        """
        if self.state != SessionState.RUNNING:
            msg = f"Cannot send messages in state: {self.state}"
            raise RuntimeError(msg)

        await self._message_queue.put(message)
        logger.debug("Message enqueued: %s", message)

    async def _validate_message(self, message: dict[str, Any]) -> bool:
        """Validate incoming message format.

        Args:
            message: Message to validate

        Returns:
            True if message is valid
        """
        # Basic structure validation
        required_keys = ["type", "content"]
        if not all(key in message for key in required_keys):
            logger.warning("Missing required keys in message: %s", message)
            return False

        # Content validation based on type
        match message["type"]:
            case "text":
                return isinstance(message["content"], str)
            case "json":
                # Validate JSON structure
                try:
                    if not isinstance(message["content"], dict):
                        return False
                except Exception:  # noqa: BLE001
                    return False
            case _:
                logger.warning("Unknown message type: %s", message["type"])
                return False

        return True

    async def close(self) -> None:
        """Close session and cleanup resources."""
        # If already closed, do nothing
        if self.state == SessionState.CLOSED:
            return

        logger.info("Closing session")

        # Only try to close if we've started
        if self._started and self.state == SessionState.RUNNING:
            self.state = SessionState.CLOSING
            try:
                # Clean up registries
                await self.processor_registry.shutdown()
                await self.tool_registry.shutdown()

                # Clear message queue
                while not self._message_queue.empty():
                    await self._message_queue.get()
                    self._message_queue.task_done()

                await self._message_queue.join()
                self.state = SessionState.CLOSED
                logger.info("Session closed successfully")

            except Exception as exc:
                logger.exception("Error during session cleanup")
                msg = "Session cleanup failed"
                raise RuntimeError(msg) from exc

    async def startup(self) -> None:
        """Initialize session resources."""
        if self._started:
            return

        try:
            # Initialize registries
            await self.processor_registry.startup()
            await self.tool_registry.startup()

            self.state = SessionState.RUNNING
            self._started = True
            logger.info("Session started successfully")

        except Exception as exc:
            logger.exception("Failed to start session")
            msg = "Session startup failed"
            raise RuntimeError(msg) from exc
