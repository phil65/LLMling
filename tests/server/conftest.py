"""Shared test fixtures for server tests."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any

from mcp.shared.memory import create_client_server_memory_streams
import pytest

from llmling.config.models import Config, GlobalSettings
from llmling.config.runtime import RuntimeConfig
from llmling.processors.registry import ProcessorRegistry
from llmling.prompts.registry import PromptRegistry
from llmling.resources import ResourceLoaderRegistry
from llmling.resources.registry import ResourceRegistry
from llmling.server import LLMLingServer
from llmling.server.mcp_inproc_session import MCPInProcSession
from llmling.testing.processors import multiply, uppercase_text
from llmling.testing.tools import analyze_ast, example_tool
from llmling.tools.registry import ToolRegistry


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest.fixture
def base_config() -> Config:
    """Create minimal test configuration."""
    return Config(
        version="1.0.0",
        global_settings=GlobalSettings(),
        resources={},
        context_processors={},
        resource_groups={},
    )


@pytest.fixture
def runtime_config(base_config: Config) -> RuntimeConfig:
    """Create test runtime configuration."""
    # Create registries first
    loader_registry = ResourceLoaderRegistry()
    processor_registry = ProcessorRegistry()

    # Create dependent registries
    resource_registry = ResourceRegistry(
        loader_registry=loader_registry,
        processor_registry=processor_registry,
    )
    prompt_registry = PromptRegistry()
    tool_registry = ToolRegistry()

    # Register default loaders
    from llmling.resources import (
        CallableResourceLoader,
        CLIResourceLoader,
        ImageResourceLoader,
        PathResourceLoader,
        SourceResourceLoader,
        TextResourceLoader,
    )

    loader_registry["text"] = TextResourceLoader
    loader_registry["path"] = PathResourceLoader
    loader_registry["cli"] = CLIResourceLoader
    loader_registry["source"] = SourceResourceLoader
    loader_registry["callable"] = CallableResourceLoader
    loader_registry["image"] = ImageResourceLoader

    # Register test processors
    processor_registry.register("multiply", multiply)
    processor_registry.register("uppercase", uppercase_text)

    # Register test tools
    tool_registry.register("example", example_tool)
    tool_registry.register("analyze", analyze_ast)

    return RuntimeConfig(
        config=base_config,
        loader_registry=loader_registry,
        processor_registry=processor_registry,
        resource_registry=resource_registry,
        prompt_registry=prompt_registry,
        tool_registry=tool_registry,
    )


@pytest.fixture
async def server(runtime_config: RuntimeConfig) -> AsyncGenerator[LLMLingServer, None]:
    """Create configured test server."""
    server = LLMLingServer(runtime=runtime_config, name="llmling-server")

    try:
        yield server
    finally:
        await server.shutdown()


@pytest.fixture
async def running_server(
    server: LLMLingServer,
) -> AsyncGenerator[tuple[LLMLingServer, tuple[Any, Any]], None]:
    """Create and start test server with memory streams."""
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        task = asyncio.create_task(
            server.server.run(
                server_streams[0],
                server_streams[1],
                server.server.create_initialization_options(),
            )
        )

        try:
            yield server, client_streams
        finally:
            task.cancel()
            await server.shutdown()


@pytest.fixture
async def client() -> MCPInProcSession:
    """Create a test client."""
    return MCPInProcSession([sys.executable, "-m", "llmling.server"])
