"""Resource loading functionality."""

from llmling.resources.base import ResourceLoader
from llmling.resources.loaders import (
    CallableResourceLoader,
    CLIResourceLoader,
    PathResourceLoader,
    SourceResourceLoader,
    TextResourceLoader,
    RepositoryResourceLoader,
    ImageResourceLoader,
)
from llmling.resources.loaders.registry import ResourceLoaderRegistry
from llmling.resources.registry import ResourceRegistry
from llmling.resources.models import LoadedResource

# Create and populate the default registry
default_registry = ResourceLoaderRegistry()
default_registry["image"] = ImageResourceLoader
default_registry["path"] = PathResourceLoader
default_registry["text"] = TextResourceLoader
default_registry["cli"] = CLIResourceLoader
default_registry["source"] = SourceResourceLoader
default_registry["callable"] = CallableResourceLoader

__all__ = [
    "CLIResourceLoader",
    "CallableResourceLoader",
    "ImageResourceLoader",
    "LoadedResource",
    "PathResourceLoader",
    "RepositoryResourceLoader",
    "ResourceLoader",
    "ResourceLoaderRegistry",
    "ResourceRegistry",
    "SourceResourceLoader",
    "TextResourceLoader",
    "default_registry",
]
