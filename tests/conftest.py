from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import anyio
from mcp import types
import pytest

from llmling.config.manager import ConfigManager
from llmling.config.models import Config, ToolConfig
from llmling.processors.registry import ProcessorRegistry
from llmling.prompts.models import ArgumentType, ExtendedPromptArgument, Prompt
from llmling.server import LLMLingServer


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest.fixture
def config_manager(test_config):
    """Get config manager with test configuration."""
    return ConfigManager(test_config)


@pytest.fixture
def processor_registry():
    """Get clean processor registry."""
    return ProcessorRegistry()


@pytest.fixture
def test_config() -> Config:
    """Create a test configuration."""
    return Config(
        version="1.0",
        contexts={},  # Empty for now
        tools={
            "example": ToolConfig(
                import_path="llmling.testing.tools.example_tool",
                name="example",
                description="Example tool for testing",
            ),
        },
    )


@pytest.fixture
def prompt_fixture(server: LLMLingServer) -> None:
    """Configure server prompt registry for testing."""
    server.session.prompt_registry.register(
        "test",
        {
            "name": "test",
            "description": "Test prompt",
            "messages": [{"role": "user", "content": "Test message"}],
        },
    )


@pytest.fixture
async def test_session(server_session_fixture: None) -> None:
    """Set up test session."""
    # The fixture will handle session configuration


@pytest.fixture
def server_session_fixture(server: LLMLingServer) -> None:
    """Configure server session for testing."""
    # Add test prompt to server's session registry
    server.session.prompt_registry.register(
        "test-prompt",
        {
            "name": "test-prompt",
            "description": "Test prompt",
            "messages": [{"role": "user", "content": "Test message"}],
        },
    )


@pytest.fixture
async def server(test_config: Config) -> AsyncGenerator[LLMLingServer, None]:
    """Create and initialize a test server."""
    server = LLMLingServer(test_config)

    # Register test prompts
    server.session.prompt_registry.register(
        "test-prompt",
        Prompt(
            name="test-prompt",
            description="Test prompt with completable arguments",
            arguments=[
                ExtendedPromptArgument(
                    name="mode",
                    description="Operation mode",
                    type=ArgumentType.ENUM,
                    enum_values=["quick", "detailed", "advanced"],
                ),
                ExtendedPromptArgument(
                    name="tool",
                    description="Tool to use",
                    type=ArgumentType.TOOL,
                    tool_names=["*"],
                ),
                ExtendedPromptArgument(
                    name="input_file",
                    description="Input file path",
                    type=ArgumentType.FILE,
                    file_patterns=["*.txt", "*.md"],
                ),
            ],
            messages=[],
        ),
    )

    # Create memory streams for the server
    send_stream, receive_stream = anyio.create_memory_object_stream[
        types.JSONRPCMessage
    ]()

    # Start the server's MCP server in the background
    server_task = asyncio.create_task(
        server.mcp_server.mcp_server.run(
            receive_stream,
            send_stream,
            server.mcp_server.mcp_server.create_initialization_options(),
        )
    )

    await server.start()
    try:
        yield server
    finally:
        server_task.cancel()
        await server.shutdown()
