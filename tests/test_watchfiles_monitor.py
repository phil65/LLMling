"""Tests for watchfiles-based file monitoring."""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import pytest

from llmling.monitors.implementations.watchfiles_watcher import WatchfilesMonitor


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
async def monitor() -> AsyncIterator[WatchfilesMonitor]:
    """Create and start a test monitor."""
    mon = WatchfilesMonitor()
    await mon.start()
    yield mon
    await mon.stop()


async def test_basic_file_watch(monitor: WatchfilesMonitor, temp_dir: Path) -> None:
    """Test basic file change detection."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("initial")

    changes: list[str] = []
    event = asyncio.Event()

    def on_change(events):
        changes.extend(normalize_path(e.path) for e in events)
        event.set()

    monitor.add_watch(test_file, callback=on_change)
    await asyncio.sleep(0.1)

    test_file.write_text("modified")

    try:
        await asyncio.wait_for(event.wait(), timeout=1.0)
        assert changes, "No changes detected"
        assert normalize_path(test_file) in changes
    except TimeoutError:
        pytest.fail(f"No changes detected for {test_file}")


async def test_pattern_matching(monitor: WatchfilesMonitor, temp_dir: Path) -> None:
    """Test pattern matching works."""
    py_file = temp_dir / "test.py"
    txt_file = temp_dir / "test.txt"

    py_file.write_text("python")
    txt_file.write_text("text")

    matched_files: list[str] = []
    event = asyncio.Event()

    def on_change(events):
        matched_files.extend(normalize_path(e.path) for e in events)
        event.set()

    monitor.add_watch(
        temp_dir,
        patterns=["*.py"],
        callback=on_change,
    )

    await asyncio.sleep(0.1)

    py_file.write_text("python modified")
    await asyncio.sleep(0.1)
    txt_file.write_text("text modified")

    try:
        await asyncio.wait_for(event.wait(), timeout=1.0)
        normalized_py_file = normalize_path(py_file)
        normalized_txt_file = normalize_path(txt_file)
        assert normalized_py_file in matched_files, "Python file change not detected"
        assert normalized_txt_file not in matched_files, "Text file was wrongly matched"
    except TimeoutError:
        pytest.fail("No changes detected")


async def test_watch_direct_file(monitor: WatchfilesMonitor, temp_dir: Path) -> None:
    """Test watching a specific file works."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("initial")

    event = asyncio.Event()
    file_changed = False

    def on_change(events):
        nonlocal file_changed
        for e in events:
            if Path(normalize_path(e.path)).name == test_file.name:
                file_changed = True
                event.set()

    monitor.add_watch(str(test_file), callback=on_change)
    await asyncio.sleep(0.1)

    test_file.write_text("modified")

    try:
        await asyncio.wait_for(event.wait(), timeout=1.0)
        assert file_changed, "File change not detected"
    except TimeoutError:
        pytest.fail("Change not detected for direct file watch")


async def test_path_resolution(monitor: WatchfilesMonitor, temp_dir: Path) -> None:
    """Test different path formats are handled correctly."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("initial")

    events_received: list[str] = []
    event = asyncio.Event()

    def on_change(events):
        events_received.extend(normalize_path(e.path) for e in events)
        event.set()

    monitor.add_watch(test_file.absolute(), callback=on_change)
    await asyncio.sleep(0.1)

    test_file.write_text("modified")

    try:
        await asyncio.wait_for(event.wait(), timeout=1.0)
        normalized_test_file = normalize_path(test_file)
        assert normalized_test_file in events_received, "File change not detected"
    except TimeoutError:
        pytest.fail("No events received")
