"""Resource loading functionality."""

from llmling.resources.base import ResourceLoader
from llmling.resources.loaders import (
    CallableResourceLoader,
    CLIResourceLoader,
    PathResourceLoader,
    SourceResourceLoader,
    TextResourceLoader,
    RepositoryResourceLoader,
)
from llmling.resources.loaders.registry import ResourceLoaderRegistry
from llmling.resources.registry import ResourceRegistry
from llmling.resources.models import LoadedResource

# Create and populate the default registry
default_registry = ResourceLoaderRegistry()
default_registry.register_default_loaders()

__all__ = [
    "CLIResourceLoader",
    "CallableResourceLoader",
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
