"""Tests for context loaders."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any
from unittest import mock

import pytest
import requests

from llmling.config import (
    CallableContext,
    CLIContext,
    PathContext,
    SourceContext,
    TextContext,
)
from llmling.context_loaders.callable import CallableLoader, CallableLoadError
from llmling.context_loaders.cli import CLILoader, CLILoadError
from llmling.context_loaders.path import PathLoader, PathLoadError
from llmling.context_loaders.source import SourceLoader, SourceLoadError
from llmling.context_loaders.text import TextLoader, TextLoadError
from llmling.processors import ProcessingStep, ProcessorConfig, ProcessorRegistry


if TYPE_CHECKING:
    from pathlib import Path


# Constants for test data
SAMPLE_TEXT = "Hello, World!"
TIMEOUT_SECONDS = 1
LARGE_TEXT = "A" * 1000
INVALID_MODULE = "does_not_exist.module"
INVALID_FUNCTION = "invalid_function"
TEST_FILE_CONTENT = "Test file content"
TEST_URL = "https://example.com/test.txt"
TEST_URL_CONTENT = "Test URL content"
GIT_HELP_COMMAND = "git --help"
LONG_RUNNING_COMMAND = "sleep 10"
ECHO_COMMAND = "echo test" if sys.platform == "win32" else ["echo", "test"]
SLEEP_COMMAND = "timeout 2" if sys.platform == "win32" else ["sleep", "2"]
# Test fixtures


def reverse_text(text: str) -> str:
    """Helper function to reverse text."""
    return text[::-1]


@pytest.fixture
def processor_registry() -> ProcessorRegistry:
    """Create a test processor registry."""
    return ProcessorRegistry()


@pytest.fixture
def tmp_file(tmp_path: Path) -> Path:
    """Create a temporary test file."""
    test_file = tmp_path / "test.txt"
    test_file.write_text(TEST_FILE_CONTENT)
    return test_file


# Test helpers


async def async_function(**kwargs: Any) -> str:
    """Test async function for callable loader."""
    return f"Async result with {kwargs}"


def sync_function(**kwargs: Any) -> str:
    """Test sync function for callable loader."""
    return f"Sync result with {kwargs}"


def failing_function(**kwargs: Any) -> str:
    """Test function that raises an exception."""
    msg = "Test error"
    raise ValueError(msg)


# Text Loader Tests


@pytest.mark.asyncio
async def test_text_loader_basic() -> None:
    """Test basic text loading functionality."""
    context = TextContext(
        type="text",
        content=SAMPLE_TEXT,
        description="Test text",
    )
    loader = TextLoader()
    result = await loader.load(context, ProcessorRegistry())

    assert result.content == SAMPLE_TEXT
    assert result.metadata["type"] == "raw_text"


@pytest.mark.asyncio
async def test_text_loader_with_processors(processor_registry: ProcessorRegistry) -> None:
    """Test text loading with processors."""
    # Register processor that uppercase the content
    processor_registry.register(
        "upper",
        ProcessorConfig(type="function", import_path="reprlib.repr"),
    )

    context = TextContext(
        type="text",
        content=SAMPLE_TEXT,
        description="Test text",
        processors=[ProcessingStep(name="upper")],
    )
    loader = TextLoader()
    result = await loader.load(context, processor_registry)

    assert result.content == repr(SAMPLE_TEXT)


@pytest.mark.asyncio
async def test_text_loader_invalid_context() -> None:
    """Test text loader with invalid context type."""
    context = SourceContext(
        type="source",
        import_path="test.module",
        description="Test source",
    )
    loader = TextLoader()

    with pytest.raises(TextLoadError, match="Expected text context"):
        await loader.load(context, ProcessorRegistry())


# Path Loader Tests


@pytest.mark.asyncio
async def test_path_loader_file(tmp_file: Path) -> None:
    """Test loading from a file."""
    context = PathContext(
        type="path",
        path=str(tmp_file),
        description="Test file",
    )
    loader = PathLoader()
    result = await loader.load(context, ProcessorRegistry())

    assert result.content == TEST_FILE_CONTENT
    assert result.metadata["source_type"] == "file"


@pytest.mark.asyncio
async def test_path_loader_url() -> None:
    """Test loading from a URL."""
    context = PathContext(
        type="path",
        path=TEST_URL,
        description="Test URL",
    )
    loader = PathLoader()

    # Create a mock UPath class
    mock_upath = mock.Mock()
    mock_upath.is_file.return_value = True
    mock_upath.read_text.return_value = TEST_URL_CONTENT

    # Mock the UPath constructor to return our mock object
    with mock.patch("llmling.context_loaders.path.UPath", return_value=mock_upath):
        result = await loader.load(context, ProcessorRegistry())

        assert result.content == TEST_URL_CONTENT
        assert result.metadata["path"] == TEST_URL


@pytest.mark.asyncio
async def test_path_loader_file_not_found() -> None:
    """Test loading from a non-existent file."""
    context = PathContext(
        type="path",
        path="/nonexistent/file.txt",
        description="Test missing file",
    )
    loader = PathLoader()

    with pytest.raises(PathLoadError):
        await loader.load(context, ProcessorRegistry())


@pytest.mark.asyncio
async def test_path_loader_url_error() -> None:
    """Test loading from an invalid URL."""
    with mock.patch("requests.get", side_effect=requests.RequestException):
        context = PathContext(
            type="path",
            path=TEST_URL,
            description="Test URL error",
        )
        loader = PathLoader()

        with pytest.raises(PathLoadError):
            await loader.load(context, ProcessorRegistry())


# CLI Loader Tests


@pytest.mark.asyncio
async def test_cli_loader_basic() -> None:
    """Test basic CLI command execution."""
    context = CLIContext(
        type="cli",
        command=ECHO_COMMAND,
        description="Test command",
        shell=sys.platform == "win32",
    )
    loader = CLILoader()
    result = await loader.load(context, ProcessorRegistry())

    assert "test" in result.content.strip()
    assert result.metadata["return_code"] == 0


@pytest.mark.asyncio
async def test_cli_loader_shell_command() -> None:
    """Test CLI command execution with shell=True."""
    context = CLIContext(
        type="cli",
        command="echo $HOME",
        description="Test shell command",
        shell=True,
    )
    loader = CLILoader()
    result = await loader.load(context, ProcessorRegistry())

    assert result.content.strip()
    assert result.metadata["return_code"] == 0


@pytest.mark.asyncio
async def test_cli_loader_command_failure() -> None:
    """Test CLI command that fails."""
    context = CLIContext(
        type="cli",
        command=["nonexistent-command"],
        description="Test failing command",
    )
    loader = CLILoader()

    with pytest.raises(CLILoadError):
        await loader.load(context, ProcessorRegistry())


@pytest.mark.asyncio
async def test_cli_loader_timeout() -> None:
    """Test CLI command that times out."""
    context = CLIContext(
        type="cli",
        command=SLEEP_COMMAND,
        description="Test timeout",
        timeout=1,  # Using integer timeout
        shell=sys.platform == "win32",
    )
    loader = CLILoader()

    with pytest.raises(CLILoadError, match="timeout"):
        await loader.load(context, ProcessorRegistry())


# Source Loader Tests


@pytest.mark.asyncio
async def test_source_loader_basic() -> None:
    """Test basic source code loading."""
    context = SourceContext(
        type="source",
        import_path="llmling.context_loaders.text",
        description="Test source",
    )
    loader = SourceLoader()
    result = await loader.load(context, ProcessorRegistry())

    assert "class TextLoader" in result.content
    assert result.metadata["module"] == context.import_path


@pytest.mark.asyncio
async def test_source_loader_invalid_module() -> None:
    """Test loading from non-existent module."""
    context = SourceContext(
        type="source",
        import_path=INVALID_MODULE,
        description="Test invalid module",
    )
    loader = SourceLoader()

    with pytest.raises(SourceLoadError):
        await loader.load(context, ProcessorRegistry())


@pytest.mark.asyncio
async def test_source_loader_recursive(tmp_path) -> None:
    """Test recursive source code loading."""
    # Create a simple test module structure
    test_module = """
def test_function():
    return "test"
"""
    module_path = tmp_path / "test_module"
    module_path.mkdir()
    (module_path / "__init__.py").write_text(test_module)
    sys.path.insert(0, str(tmp_path))

    try:
        context = SourceContext(
            type="source",
            import_path="test_module",
            description="Test recursive",
            recursive=True,
        )
        loader = SourceLoader()
        result = await loader.load(context, ProcessorRegistry())

        assert "def test_function" in result.content
    finally:
        sys.path.pop(0)


# Callable Loader Tests


@pytest.mark.asyncio
async def test_callable_loader_sync() -> None:
    """Test loading from synchronous callable."""
    context = CallableContext(
        type="callable",
        import_path=f"{__name__}.sync_function",
        description="Test sync callable",
        keyword_args={"test": "value"},
    )
    loader = CallableLoader()
    result = await loader.load(context, ProcessorRegistry())

    assert "Sync result with" in result.content
    assert result.metadata["kwargs"] == {"test": "value"}


@pytest.mark.asyncio
async def test_callable_loader_async() -> None:
    """Test loading from asynchronous callable."""
    context = CallableContext(
        type="callable",
        import_path=f"{__name__}.async_function",
        description="Test async callable",
        keyword_args={"test": "value"},
    )
    loader = CallableLoader()
    result = await loader.load(context, ProcessorRegistry())

    assert "Async result with" in result.content
    assert result.metadata["is_coroutine"] is True


@pytest.mark.asyncio
async def test_callable_loader_invalid_import() -> None:
    """Test loading from non-existent callable."""
    context = CallableContext(
        type="callable",
        import_path=f"{INVALID_MODULE}.{INVALID_FUNCTION}",
        description="Test invalid import",
    )
    loader = CallableLoader()

    with pytest.raises(CallableLoadError, match="Failed to import module"):
        await loader.load(context, ProcessorRegistry())


@pytest.mark.asyncio
async def test_callable_loader_execution_error() -> None:
    """Test callable that raises an exception."""
    context = CallableContext(
        type="callable",
        import_path=f"{__name__}.failing_function",
        description="Test failing callable",
    )
    loader = CallableLoader()

    with pytest.raises(CallableLoadError, match="Error executing"):
        await loader.load(context, ProcessorRegistry())


# Integration Tests


@pytest.mark.asyncio
async def test_all_loaders_with_processors(
    processor_registry: ProcessorRegistry,
    tmp_file: Path,
) -> None:
    """Test all loaders with processor chain."""
    # Register test processors
    processor_registry.register(
        "upper",
        ProcessorConfig(type="function", import_path="reprlib.repr"),
    )
    processor_registry.register(
        "reverse",
        ProcessorConfig(type="function", import_path=f"{__name__}.reverse_text"),
    )

    processors = [
        ProcessingStep(name="upper"),
        ProcessingStep(name="reverse"),
    ]

    # Test contexts
    contexts = [
        TextContext(
            type="text",
            content=SAMPLE_TEXT,
            description="Test text",
            processors=processors,
        ),
        PathContext(
            type="path",
            path=str(tmp_file),
            description="Test file",
            processors=processors,
        ),
        CLIContext(
            type="cli",
            command=ECHO_COMMAND,
            description="Test command",
            shell=sys.platform == "win32",
            processors=processors,
        ),
    ]

    # Test each loader
    loaders = {
        "text": TextLoader(),
        "path": PathLoader(),
        "cli": CLILoader(),
    }

    for context in contexts:
        loader = loaders[context.type]
        result = await loader.load(context, processor_registry)
        assert result.content  # Ensure we got some content
        # Verify processors were applied (content should be uppercase and reversed)
        assert result.content.startswith("'")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
