"""Tests for signal-based file monitoring."""

from __future__ import annotations

import asyncio
from pathlib import Path
import platform
import tempfile
from typing import TYPE_CHECKING

import pytest

from llmling.utils.watcher import FileWatcher


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator


def normalize_path(path: str | Path) -> str:
    """Normalize path to handle macOS /private prefix."""
    return str(Path(path).resolve())


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
async def watcher() -> AsyncIterator[FileWatcher]:
    """Create and start a test watcher."""
    w = FileWatcher()
    await w.start()
    yield w
    await w.stop()


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="File watching unreliable on macOS"
)
async def test_basic_file_watch(watcher: FileWatcher, temp_dir: Path) -> None:
    """Test basic file change detection."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("initial")

    changes: list[str] = []
    event = asyncio.Event()

    @watcher.signals.file_modified.connect
    def on_change(path: str) -> None:
        changes.append(normalize_path(path))
        event.set()

    watcher.add_watch(test_file)
    await asyncio.sleep(0.3)

    test_file.write_text("modified")

    try:
        await asyncio.wait_for(event.wait(), timeout=1.0)
        assert changes, "No changes detected"
        assert normalize_path(test_file) in changes
    except TimeoutError:
        pytest.fail(f"No changes detected for {test_file}")


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="File watching unreliable on macOS"
)
async def test_pattern_matching(watcher: FileWatcher, temp_dir: Path) -> None:
    """Test pattern matching works."""
    py_file = temp_dir / "test.py"
    txt_file = temp_dir / "test.txt"

    py_file.write_text("python")
    txt_file.write_text("text")

    matched_files: list[str] = []
    event = asyncio.Event()

    @watcher.signals.file_modified.connect
    def on_change(path: str) -> None:
        matched_files.append(normalize_path(path))
        event.set()

    watcher.add_watch(temp_dir, patterns=["*.py"])
    await asyncio.sleep(0.3)

    py_file.write_text("python modified")
    await asyncio.sleep(0.3)
    txt_file.write_text("text modified")

    try:
        await asyncio.wait_for(event.wait(), timeout=1.0)
        normalized_py_file = normalize_path(py_file)
        normalized_txt_file = normalize_path(txt_file)
        assert normalized_py_file in matched_files, "Python file change not detected"
        assert normalized_txt_file not in matched_files, "Text file was wrongly matched"
    except TimeoutError:
        pytest.fail("No changes detected")


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="File watching unreliable on macOS"
)
async def test_watch_direct_file(watcher: FileWatcher, temp_dir: Path) -> None:
    """Test watching a specific file works."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("initial")

    event = asyncio.Event()
    file_changed = False

    @watcher.signals.file_modified.connect
    def on_change(path: str) -> None:
        nonlocal file_changed
        if Path(normalize_path(path)).name == test_file.name:
            file_changed = True
            event.set()

    watcher.add_watch(str(test_file))
    await asyncio.sleep(0.3)

    test_file.write_text("modified")

    try:
        await asyncio.wait_for(event.wait(), timeout=1.0)
        assert file_changed, "File change not detected"
    except TimeoutError:
        pytest.fail("Change not detected for direct file watch")


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="File watching unreliable on macOS"
)
async def test_path_resolution(watcher: FileWatcher, temp_dir: Path) -> None:
    """Test different path formats are handled correctly."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("initial")

    events_received: list[str] = []
    event = asyncio.Event()

    @watcher.signals.file_modified.connect
    def on_change(path: str) -> None:
        events_received.append(normalize_path(path))
        event.set()

    watcher.add_watch(test_file.absolute())
    await asyncio.sleep(0.3)

    test_file.write_text("modified")

    try:
        await asyncio.wait_for(event.wait(), timeout=1.0)
        normalized_test_file = normalize_path(test_file)
        assert normalized_test_file in events_received, "File change not detected"
    except TimeoutError:
        pytest.fail("No events received")


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="File watching unreliable on macOS"
)
async def test_multiple_signals(watcher: FileWatcher, temp_dir: Path) -> None:
    """Test that all signal types work."""
    test_file = temp_dir / "test.txt"

    events: dict[str, list[str]] = {
        "added": [],
        "modified": [],
        "deleted": [],
    }
    event = asyncio.Event()

    @watcher.signals.file_added.connect
    def on_added(path: str) -> None:
        events["added"].append(normalize_path(path))
        event.set()

    @watcher.signals.file_modified.connect
    def on_modified(path: str) -> None:
        events["modified"].append(normalize_path(path))
        event.set()

    @watcher.signals.file_deleted.connect
    def on_deleted(path: str) -> None:
        events["deleted"].append(normalize_path(path))
        event.set()

    watcher.add_watch(temp_dir)
    await asyncio.sleep(0.3)

    # Test creation
    test_file.write_text("initial")
    await asyncio.wait_for(event.wait(), timeout=1.0)
    event.clear()
    assert normalize_path(test_file) in events["added"]

    # Test modification
    test_file.write_text("modified")
    await asyncio.wait_for(event.wait(), timeout=1.0)
    event.clear()
    assert normalize_path(test_file) in events["modified"]

    # Test deletion
    test_file.unlink()
    await asyncio.wait_for(event.wait(), timeout=1.0)
    assert normalize_path(test_file) in events["deleted"]


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="File watching unreliable on macOS"
)
async def test_watch_error_handling(watcher: FileWatcher, temp_dir: Path) -> None:
    """Test error handling in watcher."""
    errors: list[tuple[str, Exception]] = []
    event = asyncio.Event()

    @watcher.signals.watch_error.connect
    def on_error(path: str, exc: Exception) -> None:
        errors.append((path, exc))
        event.set()

    # Try to watch a non-existent directory
    nonexistent = temp_dir / "nonexistent"
    watcher.add_watch(nonexistent)

    try:
        await asyncio.wait_for(event.wait(), timeout=0.5)
        assert errors, "No error reported for invalid watch"
        assert str(nonexistent) in errors[0][0]
        assert isinstance(errors[0][1], FileNotFoundError)
    except TimeoutError:
        pytest.fail("No error reported within timeout")
