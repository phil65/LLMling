from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import git
import pytest

from llmling.config.models import (
    CallableResource,
    CLIResource,
    PathResource,
    RepositoryResource,
    SourceResource,
    TextResource,
)
from llmling.core import exceptions
from llmling.core.typedefs import ProcessingStep
from llmling.processors.base import ProcessorConfig
from llmling.resources import (
    CallableResourceLoader,
    CLIResourceLoader,
    PathResourceLoader,
    SourceResourceLoader,
    TextResourceLoader,
)
from llmling.resources.base import LoaderContext, create_loaded_resource
from llmling.resources.loaders.repository import RepositoryResourceLoader


if TYPE_CHECKING:
    from pathlib import Path

    from llmling.processors.registry import ProcessorRegistry
    from llmling.resources.base import ResourceLoader
    from llmling.resources.loaders.registry import ResourceLoaderRegistry


@pytest.mark.parametrize(
    ("uri", "expected_loader"),
    [
        ("text://content", TextResourceLoader),
        ("file:///path/to/file.txt", PathResourceLoader),
        ("cli://command", CLIResourceLoader),
        ("python://module.path", SourceResourceLoader),
        ("callable://func", CallableResourceLoader),
        ("repository://github.com/org/repo", RepositoryResourceLoader),
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
        (
            RepositoryResource(repo_url="https://github.com/org/repo.git"),
            RepositoryResourceLoader,
        ),
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

    result = await anext(loader.load(processor_registry=processor_registry))
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

    result = await anext(loader.load(processor_registry=processor_registry))
    assert result.content == content
    assert result.source_type == "path"


@pytest.mark.asyncio
async def test_cli_loader(processor_registry: ProcessorRegistry) -> None:
    """Test CLIResourceLoader functionality."""
    # Use a simple echo command
    resource = CLIResource(command="echo test", shell=True)
    loader = CLIResourceLoader(LoaderContext(resource=resource, name="test"))

    result = await anext(loader.load(processor_registry=processor_registry))
    assert result.content.strip() == "test"
    assert result.source_type == "cli"


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Git branch tests are unreliable in CI environment",
)
async def test_repository_loader(
    tmp_path: Path,
    processor_registry: ProcessorRegistry,
) -> None:
    """Test basic repository loading functionality."""
    # Set up a test repo
    repo = git.Repo.init(tmp_path)
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    repo.index.add(["test.txt"])
    commit = repo.index.commit("Initial commit")

    # Create and checkout main branch explicitly
    default_branch = "main"
    if default_branch not in repo.heads:
        repo.create_head(default_branch, commit)
        repo.head.reference = repo.heads[default_branch]
        repo.head.reset(
            index=True, working_tree=True
        )  # Force reset to update working tree
    else:
        repo.heads[default_branch].checkout()

    # Verify the branch is properly set up
    try:
        repo.git.rev_parse("--verify", default_branch)
    except git.exc.GitCommandError:
        pytest.skip(
            f"Cannot verify branch {default_branch} - possibly in detached HEAD state"
        )

    # Load the file
    resource = RepositoryResource(
        repo_url=str(tmp_path), ref=default_branch, path="test.txt"
    )
    loader = RepositoryResourceLoader(LoaderContext(resource=resource, name="test"))

    async for result in loader.load(processor_registry=processor_registry):
        assert result.content == "test content"
        assert result.source_type == "repository"
        assert result.metadata.mime_type == "text/plain"
        assert "repo" in result.metadata.extra
        break


@pytest.mark.asyncio
async def test_source_loader(processor_registry: ProcessorRegistry) -> None:
    """Test SourceResourceLoader functionality."""
    resource = SourceResource(import_path="llmling.core.log")
    loader = SourceResourceLoader(LoaderContext(resource=resource, name="test"))

    result = await anext(loader.load(processor_registry=processor_registry))
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
    ],
)
def test_uri_creation(uri_template: str, name: str, expected: str) -> None:
    """Test URI creation from templates."""
    test_loader = type(
        "TestLoader",
        (TextResourceLoader,),
        {"get_uri_template": staticmethod(lambda: uri_template)},
    )
    # Create instance with no context
    loader = test_loader(None)
    assert loader.create_uri(name=name) == expected  # type: ignore


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
        scheme in schemes for scheme in ["text", "file", "cli", "python", "callable"]
    )


def test_registry_uri_templates(loader_registry: ResourceLoaderRegistry) -> None:
    """Test getting URI templates from registry."""
    templates = loader_registry.get_uri_templates()
    assert len(templates) == 6  # One for each loader type  # noqa: PLR2004
    assert all("scheme" in t and "template" in t and "mimeTypes" in t for t in templates)


@pytest.mark.asyncio
async def test_path_loader_directory(
    tmp_path: Path,
    processor_registry: ProcessorRegistry,
) -> None:
    """Test PathResourceLoader with directory."""
    # Create test directory structure
    (tmp_path / "subdir").mkdir()
    (tmp_path / "file1.txt").write_text("content 1")
    (tmp_path / "file2.md").write_text("content 2")
    (tmp_path / "subdir" / "file3.txt").write_text("content 3")

    resource = PathResource(path=str(tmp_path))
    loader = PathResourceLoader(LoaderContext(resource=resource, name="test"))

    # Collect all loaded resources using async list comp
    files = [
        result async for result in loader.load(processor_registry=processor_registry)
    ]

    # Test results
    assert len(files) == 3  # noqa: PLR2004
    assert {f.content for f in files} == {"content 1", "content 2", "content 3"}
    # Test URIs use basenames
    assert all(f.metadata.uri.startswith("file:///") for f in files)
    assert {f.metadata.name for f in files} == {"file1.txt", "file2.md", "file3.txt"}
    # Test relative path metadata
    assert all("relative_to" in f.metadata.extra for f in files)
    assert str(tmp_path) == files[0].metadata.extra["relative_to"]


@pytest.mark.asyncio
async def test_path_loader_empty_directory(
    tmp_path: Path,
    processor_registry: ProcessorRegistry,
) -> None:
    """Test loading from an empty directory."""
    resource = PathResource(path=str(tmp_path))
    loader = PathResourceLoader(LoaderContext(resource=resource, name="test"))

    files = [
        result async for result in loader.load(processor_registry=processor_registry)
    ]

    assert len(files) == 0


@pytest.mark.asyncio
async def test_path_loader_directory_with_processors(
    tmp_path: Path,
    processor_registry: ProcessorRegistry,
) -> None:
    """Test directory loading with processors applied to each file."""
    # Create test files
    (tmp_path / "file1.txt").write_text("test1")
    (tmp_path / "file2.txt").write_text("test2")

    # Set up processor
    cfg = ProcessorConfig(import_path="llmling.testing.processors.reverse_text")
    processor_registry.register("reverse", cfg)
    procs = [ProcessingStep(name="reverse")]
    resource = PathResource(path=str(tmp_path), processors=procs)
    loader = PathResourceLoader(LoaderContext(resource=resource, name="test"))
    files = [
        result async for result in loader.load(processor_registry=processor_registry)
    ]

    assert len(files) == 2  # noqa: PLR2004
    assert {f.content for f in files} == {"1tset", "2tset"}  # Reversed content


def test_resource_loader_type_field():
    """Test that ResourceLoader correctly reads the type field from resources."""
    from llmling.config.models import TextResource
    from llmling.resources.loaders.text import TextResourceLoader

    loader = TextResourceLoader.create(
        resource=TextResource(name="test", content="test"), name="test"
    )

    assert loader.resource_type == "text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
