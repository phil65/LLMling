"""Resource loader implementations."""

from llmling.resources.loaders.callable import CallableResourceLoader
from llmling.resources.loaders.cli import CLIResourceLoader
from llmling.resources.loaders.path import PathResourceLoader
from llmling.resources.loaders.source import SourceResourceLoader
from llmling.resources.loaders.text import TextResourceLoader
from llmling.resources.loaders.image import ImageResourceLoader
from llmling.resources.loaders.registry import ResourceLoaderRegistry

__all__ = [
    "CLIResourceLoader",
    "CallableResourceLoader",
    "ImageResourceLoader",
    "PathResourceLoader",
    "ResourceLoaderRegistry",
    "SourceResourceLoader",
    "TextResourceLoader",
]
