from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import pytest

from llmling.config.models import PathResource, TextResource, WatchConfig
from llmling.processors.registry import ProcessorRegistry
from llmling.resources import ResourceLoaderRegistry
from llmling.resources.registry import ResourceRegistry


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest.fixture
async def resource_registry() -> AsyncGenerator[ResourceRegistry, None]:
    """Create a test resource registry."""
    registry = ResourceRegistry(
        loader_registry=ResourceLoaderRegistry(),
        processor_registry=ProcessorRegistry(),
    )
    await registry.startup()
    yield registry
    await registry.shutdown()


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


async def test_watch_enabled(resource_registry: ResourceRegistry, temp_dir: Path) -> None:
    """Test that watching can be enabled for a resource."""
    # Create test file
    test_file = temp_dir / "test.txt"
    test_file.write_text("initial")

    # Create watched resource
    resource = PathResource(
        path=str(test_file),
        watch=WatchConfig(enabled=True),
    )

    # Register and verify watch is set up
    resource_registry.register("test", resource)
    assert "test" in resource_registry.watcher.handlers

    # Modify file and wait for notification
    event = asyncio.Event()
    original_invalidate = resource_registry.invalidate

    def on_invalidate(name: str) -> None:
        original_invalidate(name)
        event.set()

    resource_registry.invalidate = on_invalidate  # type: ignore

    # Change file
    test_file.write_text("modified")

    # Wait for notification
    try:
        await asyncio.wait_for(event.wait(), timeout=2.0)
    except TimeoutError:
        pytest.fail("Timeout waiting for file change notification")


async def test_watch_disabled(
    resource_registry: ResourceRegistry, temp_dir: Path
) -> None:
    """Test that watching can be disabled."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("initial")

    # Create unwatched resource
    resource = PathResource(
        path=str(test_file),
        watch=WatchConfig(enabled=False),
    )

    resource_registry.register("test", resource)
    assert "test" not in resource_registry.watcher.handlers


# async def test_watch_patterns(
#     resource_registry: ResourceRegistry, temp_dir: Path
# ) -> None:
#     """Test watch patterns are respected."""
#     # Create test files
#     (temp_dir / "test.py").write_text("python")
#     (temp_dir / "test.txt").write_text("text")

#     # Create watched resource with pattern
#     # Only watch Python files
#     cfg = WatchConfig(enabled=True, patterns=["*.py"])
#     resource = PathResource(path=str(temp_dir), watch=cfg)
#     resource_registry.register("test", resource)
#     # Set up event tracking
#     events = []
#     original_invalidate = resource_registry.invalidate

#     def on_invalidate(name: str) -> None:
#         original_invalidate(name)
#         events.append(name)

#     resource_registry.invalidate = on_invalidate  # type: ignore

#     # Modify both files
#     (temp_dir / "test.py").write_text("python modified")
#     await asyncio.sleep(0.5)  # Let watchdog process
#     (temp_dir / "test.txt").write_text("text modified")
#     await asyncio.sleep(0.5)  # Let watchdog process

#     # Should only get one event for the Python file
#     assert len(events) == 1
#     assert "test" in events


async def test_watch_cleanup(resource_registry: ResourceRegistry, temp_dir: Path) -> None:
    """Test that watches are cleaned up properly."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("initial")

    # Create and register watched resource
    resource = PathResource(
        path=str(test_file),
        watch=WatchConfig(enabled=True),
    )

    resource_registry.register("test", resource)
    assert "test" in resource_registry.watcher.handlers

    # Delete resource and verify watch is removed
    del resource_registry["test"]
    assert "test" not in resource_registry.watcher.handlers


async def test_supports_watching(temp_dir: Path) -> None:
    """Test that supports_watching property works correctly."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("test")

    # Path resource should support watching for existing files
    path_resource = PathResource(path=str(test_file))
    assert path_resource.supports_watching

    # Non-existent paths should not support watching
    nonexistent = PathResource(path="/nonexistent/path")
    assert not nonexistent.supports_watching

    # Text resources never support watching
    text_resource = TextResource(content="some text")
    assert not text_resource.supports_watching


async def test_watch_invalid_path(resource_registry: ResourceRegistry) -> None:
    """Test handling of invalid paths."""
    # Create resource with non-existent path
    resource = PathResource(
        path="/nonexistent/path",
        watch=WatchConfig(enabled=True),
    )

    # Should register but log warning about invalid path
    with pytest.warns(UserWarning, match="Cannot watch non-existent path"):
        resource_registry.register("test", resource)


if __name__ == "__main__":
    pytest.main([__file__])
