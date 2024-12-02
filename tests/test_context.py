"""Tests for context loaders."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import pytest
import upath

from llmling.config.models import (
    CallableResource,
    CLIResource,
    PathResource,
    Resource,
    SourceResource,
    TextResource,
)
from llmling.core import exceptions
from llmling.core.typedefs import ProcessingStep
from llmling.processors.base import ProcessorConfig
from llmling.processors.registry import ProcessorRegistry
from llmling.resources.loaders import (
    CallableResourceLoader,
    CLIResourceLoader,
    PathResourceLoader,
    SourceResourceLoader,
    TextResourceLoader,
)
from llmling.resources.models import LoadedResource
from tests.test_processors import REVERSED_TEXT


if TYPE_CHECKING:
    from pathlib import Path

    from llmling.resources.base import ResourceLoader

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


def reverse_text(text: str) -> str:
    """Helper function to reverse text."""
    return text[::-1]


# Fixtures


@pytest.fixture
def tmp_file(tmp_path: Path) -> Path:
    """Create a temporary test file."""
    test_file = tmp_path / "test.txt"
    test_file.write_text(TEST_FILE_CONTENT)
    return test_file


# Text Loader Tests
@pytest.mark.asyncio
async def test_text_loader_basic() -> None:
    """Test basic text loading functionality."""
    context = TextResource(content=SAMPLE_TEXT, description="Test text")
    loader = TextResourceLoader()
    result = await anext(loader.load(context, ProcessorRegistry()))

    assert isinstance(result, LoadedResource)
    assert result.content == SAMPLE_TEXT
    assert result.metadata.extra["type"] == "text"


@pytest.mark.asyncio
async def test_text_loader_with_processors(processor_registry: ProcessorRegistry) -> None:
    """Test text loading with processors."""
    await processor_registry.startup()
    try:
        path = "llmling.testing.processors.reverse_text"
        cfg = ProcessorConfig(type="function", import_path=path)
        processor_registry.register("reverse", cfg)
        steps = [ProcessingStep(name="reverse")]
        context = TextResource(content=SAMPLE_TEXT, description="test", processors=steps)
        loader = TextResourceLoader()
        result = await anext(loader.load(context, processor_registry))
        assert result.content == REVERSED_TEXT
    finally:
        await processor_registry.shutdown()


# Path Loader Tests
@pytest.mark.asyncio
async def test_path_loader_file(tmp_file: Path) -> None:
    """Test loading from a file."""
    context = PathResource(path=str(tmp_file), description="Test file")
    loader = PathResourceLoader()
    coro = loader.load(context, ProcessorRegistry())
    result = await anext(coro)

    assert result.content == TEST_FILE_CONTENT
    assert result.metadata.extra["type"] == "path"
    assert result.metadata.extra["path"] == str(tmp_file)


@pytest.mark.asyncio
async def test_path_loader_with_file_protocol(tmp_path: Path) -> None:
    """Test loading from a path with file:// protocol."""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text(TEST_FILE_CONTENT)

    # Use UPath to create the proper file:// URL
    path = upath.UPath(test_file)
    file_url = str(path.as_uri())  # This will create the correct file:// URL format

    context = PathResource(path=file_url, description="Test file URL")

    loader = PathResourceLoader()
    result = await anext(loader.load(context, ProcessorRegistry()))

    assert result.content == TEST_FILE_CONTENT
    assert result.metadata.extra["path"] == file_url
    assert result.metadata.extra["scheme"] == "file"
    assert result.metadata.size == len(TEST_FILE_CONTENT)


@pytest.mark.asyncio
async def test_path_loader_error() -> None:
    """Test loading from a non-existent path."""
    context = PathResource(path="/nonexistent/file.txt", description="Test missing file")
    loader = PathResourceLoader()

    with pytest.raises(exceptions.LoaderError):
        await anext(loader.load(context, ProcessorRegistry()))


# CLI Loader Tests
@pytest.mark.asyncio
async def test_cli_loader_basic() -> None:
    """Test basic CLI command execution."""
    is_shell = sys.platform == "win32"
    context = CLIResource(
        command=ECHO_COMMAND, description="Test command", shell=is_shell
    )
    loader = CLIResourceLoader()
    result = await anext(loader.load(context, ProcessorRegistry()))

    assert "test" in result.content.strip()
    assert result.metadata.extra["exit_code"] == 0


@pytest.mark.asyncio
async def test_cli_loader_timeout() -> None:
    """Test CLI command timeout."""
    context = CLIResource(command=SLEEP_COMMAND, timeout=0.1, description="test")
    loader = CLIResourceLoader()

    with pytest.raises(exceptions.LoaderError):
        await anext(loader.load(context, ProcessorRegistry()))


# Source Loader Tests
@pytest.mark.asyncio
async def test_source_loader_basic() -> None:
    """Test basic source code loading."""
    path = "llmling.resources.loaders.text"
    context = SourceResource(import_path=path, description="Test source")
    loader = SourceResourceLoader()
    result = await anext(loader.load(context, ProcessorRegistry()))

    assert "class TextResourceLoader" in result.content
    assert result.metadata.extra["import_path"] == context.import_path


@pytest.mark.asyncio
async def test_source_loader_invalid_module() -> None:
    """Test loading from non-existent module."""
    context = SourceResource(
        import_path=INVALID_MODULE, description="Test invalid module"
    )
    loader = SourceResourceLoader()

    with pytest.raises(exceptions.LoaderError):
        await anext(loader.load(context, ProcessorRegistry()))


# Callable Loader Tests
@pytest.mark.asyncio
async def test_callable_loader_sync() -> None:
    """Test loading from synchronous callable."""
    context = CallableResource(
        import_path=f"{__name__}.sync_function",
        description="Test sync callable",
        keyword_args={"test": "value"},
    )
    loader = CallableResourceLoader()
    result = await anext(loader.load(context, processor_registry=ProcessorRegistry()))

    assert "Sync result with" in result.content
    assert result.metadata.extra["import_path"] == context.import_path


@pytest.mark.asyncio
async def test_callable_loader_async() -> None:
    """Test loading from asynchronous callable."""
    context = CallableResource(
        import_path=f"{__name__}.async_function",
        description="Test async callable",
        keyword_args={"test": "value"},
    )
    loader = CallableResourceLoader()
    result = await anext(loader.load(context, ProcessorRegistry()))

    assert "Async result with" in result.content
    assert result.metadata.extra["import_path"] == context.import_path


# Integration Tests
@pytest.mark.asyncio
async def test_all_loaders_with_processors(
    processor_registry: ProcessorRegistry,
    tmp_file: Path,
) -> None:
    """Test all loaders with processor chain."""
    cfg = ProcessorConfig(type="function", import_path="reprlib.repr")
    processor_registry.register("upper", cfg)
    cfg = ProcessorConfig(type="function", import_path=f"{__name__}.reverse_text")
    processor_registry.register("reverse", cfg)
    processors = [ProcessingStep(name="upper"), ProcessingStep(name="reverse")]

    resources: list[Resource] = [
        TextResource(content=SAMPLE_TEXT, description="Test text", processors=processors),
        PathResource(path=str(tmp_file), description="Test file", processors=processors),
        CLIResource(
            command=ECHO_COMMAND,
            description="Test command",
            shell=sys.platform == "win32",
            processors=processors,
        ),
    ]

    loaders: dict[str, ResourceLoader[Any]] = {
        "text": TextResourceLoader(),
        "path": PathResourceLoader(),
        "cli": CLIResourceLoader(),
    }

    for context in resources:
        loader = loaders[context.resource_type]
        result = await anext(loader.load(context, processor_registry))
        assert isinstance(result, LoadedResource)
        assert result.content
        assert result.content.startswith("'")


if __name__ == "__main__":
    pytest.main(["-vv", __file__])
