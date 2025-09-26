from __future__ import annotations

import asyncio
from pathlib import Path
import platform
import tempfile
from typing import TYPE_CHECKING
import warnings

import pytest

from llmling.config.models import PathResource, TextResource, WatchConfig
from llmling.core.log import get_logger
from llmling.processors.registry import ProcessorRegistry
from llmling.resources import ResourceLoaderRegistry
from llmling.resources.registry import ResourceRegistry


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator

logger = get_logger(__name__)


@pytest.fixture
async def resource_registry() -> AsyncIterator[ResourceRegistry]:
    """Create a test resource registry."""
    loader_registry = ResourceLoaderRegistry()
    # Explicitly register the path loader
    from llmling.resources.loaders.path import PathResourceLoader

    loader_registry["path"] = PathResourceLoader

    registry = ResourceRegistry(
        loader_registry=loader_registry,
        processor_registry=ProcessorRegistry(),
    )
    await registry.startup()
    yield registry
    await registry.shutdown()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="File watching unreliable on macOS"
)
async def test_watch_enabled(resource_registry: ResourceRegistry, temp_dir: Path) -> None:
    """Test that watching can be enabled for a resource."""
    # Create test file
    test_file = temp_dir / "test.txt"
    test_file.write_text("initial")
    test_file.touch()

    # Create watched resource
    resource = PathResource(path=str(test_file), watch=WatchConfig(enabled=True))

    # Track invalidations
    event = asyncio.Event()
    invalidated_resources: list[str] = []

    def on_invalidate(name: str) -> None:
        invalidated_resources.append(name)
        event.set()

    # Connect to invalidation via monkey patch
    resource_registry.invalidate = on_invalidate  # type: ignore

    # Register resource
    resource_registry.register("test", resource)

    # Small delay to ensure watch is set up
    await asyncio.sleep(0.1)

    # Modify file
    test_file.write_text("modified")
    test_file.touch()

    try:
        await asyncio.wait_for(event.wait(), timeout=2.0)
        assert "test" in invalidated_resources
    except TimeoutError:
        pytest.fail("Timeout waiting for file change signal")


async def test_watch_disabled(
    resource_registry: ResourceRegistry, temp_dir: Path
) -> None:
    """Test that watching can be disabled."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("initial")

    # Create unwatched resource
    resource = PathResource(path=str(test_file), watch=WatchConfig(enabled=False))

    # Track invalidations
    event = asyncio.Event()

    def on_invalidate(name: str) -> None:
        event.set()

    resource_registry.invalidate = on_invalidate  # type: ignore
    resource_registry.register("test", resource)

    # Modify file
    test_file.write_text("modified")
    test_file.touch()

    try:
        await asyncio.wait_for(event.wait(), timeout=0.5)
        pytest.fail("Received unexpected file change signal")
    except TimeoutError:
        pass  # Expected - no signal should be emitted


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="File watching unreliable on macOS"
)
async def test_watch_patterns(
    resource_registry: ResourceRegistry, temp_dir: Path
) -> None:
    """Test watch patterns are respected."""
    py_file = temp_dir / "test.py"
    txt_file = temp_dir / "test.txt"

    py_file.write_text("python")
    txt_file.write_text("text")

    # Create watched resource with pattern
    cfg = WatchConfig(enabled=True, patterns=["*.py"])
    resource = PathResource(path=str(temp_dir), watch=cfg)

    # Track invalidations
    event = asyncio.Event()
    invalidated_resources: list[str] = []

    def on_invalidate(name: str) -> None:
        logger.debug("Resource invalidated: %s", name)
        invalidated_resources.append(name)
        event.set()

    resource_registry.invalidate = on_invalidate  # type: ignore
    resource_registry.register("test", resource)

    # Give watcher time to set up
    await asyncio.sleep(0.5)  # Increased delay

    # Modify python file with explicit sync
    py_file.write_text("python modified")

    try:
        await asyncio.wait_for(event.wait(), timeout=2.0)
        assert "test" in invalidated_resources, "Python file change not detected"
        event.clear()
        invalidated_resources.clear()

        # Modify txt file - should not trigger
        txt_file.write_text("text modified")
        txt_file.touch()

        try:
            await asyncio.wait_for(event.wait(), timeout=0.5)
            pytest.fail("Received unexpected notification for .txt file")
        except TimeoutError:
            pass  # Expected

    except TimeoutError:
        pytest.fail("Timeout waiting for Python file change signal")


async def test_watch_cleanup(resource_registry: ResourceRegistry, temp_dir: Path) -> None:
    """Test that watches are cleaned up properly."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("initial")

    resource = PathResource(path=str(test_file), watch=WatchConfig(enabled=True))

    # Track events after cleanup
    event = asyncio.Event()

    def on_invalidate(name: str) -> None:
        event.set()

    resource_registry.invalidate = on_invalidate  # type: ignore
    resource_registry.register("test", resource)

    # Remove resource
    del resource_registry["test"]

    # Modify file - should not trigger event
    test_file.write_text("modified")
    test_file.touch()

    try:
        await asyncio.wait_for(event.wait(), timeout=0.5)
        pytest.fail("Received event after cleanup")
    except TimeoutError:
        pass  # Expected


# These tests don't need changes as they don't involve the watcher
async def test_supports_watching(temp_dir: Path) -> None:
    """Test that supports_watching property works correctly."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("test")

    path_resource = PathResource(path=str(test_file))
    assert path_resource.supports_watching

    nonexistent = PathResource(path="/nonexistent/path")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert not nonexistent.supports_watching

    text_resource = TextResource(content="some text")
    assert not text_resource.supports_watching


async def test_watch_invalid_path(resource_registry: ResourceRegistry) -> None:
    """Test handling of invalid paths."""
    resource = PathResource(path="/nonexistent/path", watch=WatchConfig(enabled=True))

    with pytest.warns(UserWarning, match="Cannot watch non-existent path"):
        resource_registry.register("test", resource)
