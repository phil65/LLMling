"""Shared test fixtures for server tests."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any

from mcp.shared.memory import create_client_server_memory_streams
import pytest

from llmling.config.models import Config, GlobalSettings
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
def resource_registry(loader_registry, processor_registry) -> ResourceRegistry:
    """Create test resource registry."""
    return ResourceRegistry(
        loader_registry=loader_registry, processor_registry=processor_registry
    )


@pytest.fixture
def loader_registry() -> ResourceLoaderRegistry:
    """Create test resource loader registry."""
    registry = ResourceLoaderRegistry()
    # Register standard loaders
    from llmling.resources import (
        CallableResourceLoader,
        CLIResourceLoader,
        ImageResourceLoader,
        PathResourceLoader,
        SourceResourceLoader,
        TextResourceLoader,
    )

    registry["text"] = TextResourceLoader
    registry["path"] = PathResourceLoader
    registry["cli"] = CLIResourceLoader
    registry["source"] = SourceResourceLoader
    registry["callable"] = CallableResourceLoader
    registry["image"] = ImageResourceLoader
    return registry


@pytest.fixture
def processor_registry() -> ProcessorRegistry:
    """Create test processor registry."""
    registry = ProcessorRegistry()
    # Register test processors
    registry.register("multiply", multiply)
    registry.register("uppercase", uppercase_text)
    return registry


@pytest.fixture
def prompt_registry() -> PromptRegistry:
    """Create test prompt registry."""
    return PromptRegistry()


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Create test tool registry."""
    registry = ToolRegistry()
    # Register test tools
    registry.register("example", example_tool)
    registry.register("analyze", analyze_ast)
    return registry


@pytest.fixture
async def server(
    base_config: Config,
    loader_registry: ResourceLoaderRegistry,
    processor_registry: ProcessorRegistry,
    prompt_registry: PromptRegistry,
    tool_registry: ToolRegistry,
) -> AsyncGenerator[LLMLingServer, None]:
    """Create configured test server."""
    server = LLMLingServer(
        config=base_config,
        name="test-server",
        loader_registry=loader_registry,
        processor_registry=processor_registry,
        prompt_registry=prompt_registry,
        tool_registry=tool_registry,
    )

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
