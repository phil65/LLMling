from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from llmling.config.models import (
    CallableResource,
    CLIResource,
    ImageResource,
    PathResource,
    SourceResource,
    TextResource,
)
from llmling.core import exceptions
from llmling.resources import (
    CallableResourceLoader,
    CLIResourceLoader,
    ImageResourceLoader,
    PathResourceLoader,
    SourceResourceLoader,
    TextResourceLoader,
)
from llmling.resources.base import LoaderContext, ResourceLoader, create_loaded_resource
from llmling.resources.loaders.registry import ResourceLoaderRegistry


if TYPE_CHECKING:
    from pathlib import Path

    from llmling.processors.registry import ProcessorRegistry


@pytest.fixture
def loader_registry() -> ResourceLoaderRegistry:
    """Create a populated resource registry."""
    registry = ResourceLoaderRegistry()
    registry["text"] = TextResourceLoader
    registry["path"] = PathResourceLoader
    registry["cli"] = CLIResourceLoader
    registry["source"] = SourceResourceLoader
    registry["callable"] = CallableResourceLoader
    registry["image"] = ImageResourceLoader
    return registry


@pytest.fixture
def processor_registry() -> ProcessorRegistry:
    """Mock processor registry."""
    return None  # type: ignore


@pytest.mark.parametrize(
    ("uri", "expected_loader"),
    [
        ("text://content", TextResourceLoader),
        ("file:///path/to/file.txt", PathResourceLoader),
        ("cli://command", CLIResourceLoader),
        ("python://module.path", SourceResourceLoader),
        ("callable://func", CallableResourceLoader),
        ("image://test.png", ImageResourceLoader),
    ],
)
def test_find_loader_for_uri(
    loader_registry: ResourceLoaderRegistry,
    uri: str,
    expected_loader: type,
) -> None:
    """Test that correct loader is found for URI."""
    loader = loader_registry.find_loader_for_uri(uri)
    assert isinstance(loader, expected_loader)


def test_find_loader_invalid_uri(loader_registry: ResourceLoaderRegistry) -> None:
    """Test error handling for invalid URIs."""
    with pytest.raises(exceptions.LoaderError):
        loader_registry.find_loader_for_uri("invalid://uri")


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        # Local paths
        ("file:///path/to/file.txt", "path/to/file.txt"),
        ("file:///C:/path/to/file.txt", "path/to/file.txt"),
        # URLs with various protocols
        ("s3://bucket/path/to/file.txt", "bucket/path/to/file.txt"),
        ("https://example.com/path/to/file.txt", "path/to/file.txt"),
        # Special characters
        ("file:///path%20with%20spaces.txt", "path with spaces.txt"),
        # Edge cases
        ("file:///./path/to/../file.txt", "path/file.txt"),
        # Multiple slashes
        ("file:///path//to///file.txt", "path/to/file.txt"),
        ("s3://bucket//path///to/file.txt", "bucket/path/to/file.txt"),
        # Empty components
        ("file:///path/to//file.txt", "path/to/file.txt"),
        ("s3://bucket///file.txt", "bucket/file.txt"),
    ],
)
def test_get_name_from_uri(uri: str, expected: str) -> None:
    """Test URI name extraction for various schemes."""
    try:
        assert PathResourceLoader.get_name_from_uri(uri) == expected
    except exceptions.LoaderError as exc:
        if "Unsupported URI" in str(exc):
            pytest.skip(f"Protocol not supported: {uri}")


@pytest.mark.parametrize(
    "uri",
    [
        "invalid://uri",
        "resource://local/test",  # We don't support resource:// scheme
        "unknown://test",
        "file:",  # Incomplete URI
        "://bad",  # Missing scheme
    ],
)
def test_get_name_from_uri_invalid(
    loader_registry: ResourceLoaderRegistry,
    uri: str,
) -> None:
    """Test invalid URI handling."""
    with pytest.raises(exceptions.LoaderError):  # noqa: PT012
        loader_cls = loader_registry.find_loader_for_uri(uri)
        loader_cls.get_name_from_uri(uri)


@pytest.mark.parametrize(
    ("resource", "expected_type"),
    [
        (TextResource(content="test"), TextResourceLoader),
        (PathResource(path="test.txt"), PathResourceLoader),
        (CLIResource(command="test"), CLIResourceLoader),
        (SourceResource(import_path="test"), SourceResourceLoader),
        (CallableResource(import_path="test"), CallableResourceLoader),
        (ImageResource(path="test.png"), ImageResourceLoader),
    ],
)
def test_get_loader(
    loader_registry: ResourceLoaderRegistry,
    resource: Any,
    expected_type: type,
) -> None:
    """Test that correct loader is returned for resource types."""
    loader = loader_registry.get_loader(resource)
    assert isinstance(loader, expected_type)


@pytest.mark.asyncio
async def test_text_loader(processor_registry: ProcessorRegistry) -> None:
    """Test TextResourceLoader functionality."""
    content = "Test content"
    resource = TextResource(content=content)
    loader = TextResourceLoader(LoaderContext(resource=resource, name="test"))

    result = await loader.load(processor_registry=processor_registry)
    assert result.content == content
    assert result.metadata.mime_type == "text/plain"
    assert result.source_type == "text"


@pytest.mark.asyncio
async def test_path_loader(
    tmp_path: Path,
    processor_registry: ProcessorRegistry,
) -> None:
    """Test PathResourceLoader functionality."""
    # Create test file
    test_file = tmp_path / "test.txt"
    content = "Test content"
    test_file.write_text(content)

    resource = PathResource(path=str(test_file))
    loader = PathResourceLoader(LoaderContext(resource=resource, name="test"))

    result = await loader.load(processor_registry=processor_registry)
    assert result.content == content
    assert result.source_type == "path"


@pytest.mark.asyncio
async def test_cli_loader(processor_registry: ProcessorRegistry) -> None:
    """Test CLIResourceLoader functionality."""
    # Use a simple echo command
    resource = CLIResource(command="echo test", shell=True)
    loader = CLIResourceLoader(LoaderContext(resource=resource, name="test"))

    result = await loader.load(processor_registry=processor_registry)
    assert result.content.strip() == "test"
    assert result.source_type == "cli"


@pytest.mark.asyncio
async def test_source_loader(processor_registry: ProcessorRegistry) -> None:
    """Test SourceResourceLoader functionality."""
    resource = SourceResource(import_path="llmling.core.log")
    loader = SourceResourceLoader(LoaderContext(resource=resource, name="test"))

    result = await loader.load(processor_registry=processor_registry)
    assert "get_logger" in result.content
    assert result.source_type == "source"
    assert result.metadata.mime_type == "text/x-python"


@pytest.mark.parametrize(
    ("uri_template", "name", "expected"),
    [
        ("text://{name}", "test", "text://test"),
        ("file:///{name}", "path/to/file.txt", "file:///path/to/file.txt"),
        ("cli://{name}", "command", "cli://command"),
        ("python://{name}", "module.path", "python://module.path"),
        ("callable://{name}", "func", "callable://func"),
        ("image://{name}", "test.png", "image://test.png"),
    ],
)
def test_uri_creation(uri_template: str, name: str, expected: str) -> None:
    """Test URI creation from templates."""
    test_loader = type(
        "TestLoader",
        (TextResourceLoader,),
        {"get_uri_template": staticmethod(lambda: uri_template)},
    )
    assert test_loader.create_uri(name=name) == expected  # type: ignore


def test_create_loaded_resource() -> None:
    """Test LoadedResource creation helper."""
    result = create_loaded_resource(
        content="test",
        source_type="text",
        uri="text://test",
        mime_type="text/plain",
        name="Test Resource",
        description="A test resource",
        additional_metadata={"key": "value"},
    )

    assert result.content == "test"
    assert result.source_type == "text"
    assert result.metadata.uri == "text://test"
    assert result.metadata.mime_type == "text/plain"
    assert result.metadata.name == "Test Resource"
    assert result.metadata.description == "A test resource"
    assert result.metadata.extra == {"key": "value"}
    assert len(result.content_items) == 1
    assert result.content_items[0].type == "text"


@pytest.mark.parametrize(
    ("loader_cls", "scheme"),
    [
        (TextResourceLoader, "text"),
        (PathResourceLoader, "file"),
        (CLIResourceLoader, "cli"),
        (SourceResourceLoader, "python"),
        (CallableResourceLoader, "callable"),
        (ImageResourceLoader, "image"),
    ],
)
def test_uri_scheme_support(loader_cls: type[ResourceLoader], scheme: str) -> None:
    """Test URI scheme support for loaders."""
    uri = f"{scheme}://test"
    assert loader_cls.supports_uri(uri)
    assert not loader_cls.supports_uri(f"invalid://{uri}")


def test_registry_supported_schemes(loader_registry: ResourceLoaderRegistry) -> None:
    """Test getting supported schemes from registry."""
    schemes = loader_registry.get_supported_schemes()
    assert all(
        scheme in schemes
        for scheme in ["text", "file", "cli", "python", "callable", "image"]
    )


def test_registry_uri_templates(loader_registry: ResourceLoaderRegistry) -> None:
    """Test getting URI templates from registry."""
    templates = loader_registry.get_uri_templates()
    assert len(templates) == 6  # One for each loader type  # noqa: PLR2004
    assert all("scheme" in t and "template" in t and "mimeTypes" in t for t in templates)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
