"""Context loading functionality."""

from llmling.context.base import ContextLoader
from llmling.context.loaders import (
    CallableContextLoader,
    CLIContextLoader,
    PathContextLoader,
    SourceContextLoader,
    TextContextLoader,
)
from llmling.context.registry import ContextLoaderRegistry
from llmling.context.models import LoadedContext
from llmling.context.loaders.image import ImageContextLoader

# Create and populate the default registry
default_registry = ContextLoaderRegistry()
default_registry.register("image", ImageContextLoader)
default_registry.register("path", PathContextLoader)
default_registry.register("text", TextContextLoader)
default_registry.register("cli", CLIContextLoader)
default_registry.register("source", SourceContextLoader)
default_registry.register("callable", CallableContextLoader)

__all__ = [
    "ContextLoader",
    "LoadedContext",
    "ContextLoaderRegistry",
    "default_registry",
    "CallableContextLoader",
    "CLIContextLoader",
    "PathContextLoader",
    "SourceContextLoader",
    "TextContextLoader",
]
