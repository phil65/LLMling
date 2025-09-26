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


if TYPE_CHECKING:
    from pathlib import Path

    from llmling.config.models import BaseResource
    from llmling.resources.base import ResourceLoader

# Constants for test data
SAMPLE_TEXT = "Hello, World!"
REVERSED_TEXT = SAMPLE_TEXT[::-1]
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
        cfg = ProcessorConfig(import_path=path)
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
    ctx = SourceResource(import_path=INVALID_MODULE, description="Test invalid module")
    loader = SourceResourceLoader()

    with pytest.raises(exceptions.LoaderError):
        await anext(loader.load(ctx, ProcessorRegistry()))


# Callable Loader Tests
@pytest.mark.asyncio
async def test_callable_loader_sync() -> None:
    """Test loading from synchronous callable."""
    context = CallableResource(
        import_path="llmling.testing.processors.multiply",
        description="Test sync callable",
        keyword_args={"text": "test", "times": 2},
    )
    loader = CallableResourceLoader()
    result = await anext(loader.load(context, processor_registry=ProcessorRegistry()))

    assert result.content == "test" * 2
    assert result.metadata.extra["import_path"] == context.import_path


@pytest.mark.asyncio
async def test_callable_loader_async() -> None:
    """Test loading from asynchronous callable."""
    context = CallableResource(
        import_path="llmling.testing.processors.async_reverse_text",
        description="Test async callable",
        keyword_args={"text": "test"},
    )
    loader = CallableResourceLoader()
    result = await anext(loader.load(context, ProcessorRegistry()))

    assert result.content == "tset"
    assert result.metadata.extra["import_path"] == context.import_path


# Integration Tests
@pytest.mark.asyncio
async def test_all_loaders_with_processors(
    processor_registry: ProcessorRegistry,
    tmp_file: Path,
) -> None:
    """Test all loaders with processor chain."""
    cfg = ProcessorConfig(import_path="llmling.testing.processors.uppercase_text")
    processor_registry.register("upper", cfg)
    cfg = ProcessorConfig(import_path="llmling.testing.processors.reverse_text")
    processor_registry.register("reverse", cfg)

    processors = [ProcessingStep(name="upper"), ProcessingStep(name="reverse")]

    # Create test content
    text_content = "Hello, World!"
    file_content = "Test file content"
    tmp_file.write_text(file_content)

    # Use the actual ECHO_COMMAND content
    echo_output = "test"  # 'echo test' command outputs just 'test'
    text_resource = TextResource(
        content=text_content,
        description="Test text",
        processors=processors,
    )

    path_resource = PathResource(
        path=str(tmp_file),
        description="Test file",
        processors=processors,
    )

    cli_resource = CLIResource(
        command=ECHO_COMMAND,
        description="Test command",
        shell=sys.platform == "win32",
        processors=processors,
    )

    # Create list of resources with their expected original content
    resources: list[tuple[BaseResource, str]] = [
        (text_resource, text_content),
        (path_resource, file_content),
        (cli_resource, echo_output),
    ]

    loaders: dict[str, ResourceLoader[Any]] = {
        "text": TextResourceLoader(),
        "path": PathResourceLoader(),
        "cli": CLIResourceLoader(),
    }

    for context, original_content in resources:
        loader = loaders[context.type]
        result = await anext(loader.load(context, processor_registry))
        assert isinstance(result, LoadedResource)
        assert result.content
        # First uppercase, then reverse
        expected = original_content.strip().upper()[::-1]
        assert result.content.strip() == expected


if __name__ == "__main__":
    pytest.main(["-vv", __file__])
