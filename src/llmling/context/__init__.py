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
default_registry["image"] = ImageContextLoader
default_registry["path"] = PathContextLoader
default_registry["text"] = TextContextLoader
default_registry["cli"] = CLIContextLoader
default_registry["source"] = SourceContextLoader
default_registry["callable"] = CallableContextLoader

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
