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
default_registry.register(ImageContextLoader)
default_registry.register(PathContextLoader)
default_registry.register(TextContextLoader)
default_registry.register(CLIContextLoader)
default_registry.register(SourceContextLoader)
default_registry.register(CallableContextLoader)

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
