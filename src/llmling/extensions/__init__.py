"""Extension system for loading MCP components from entry points."""

from llmling.extensions.base import BaseExtensionLoader
from llmling.extensions.loaders import ToolsetLoader


__all__ = [
    "BaseExtensionLoader",
    "ToolsetLoader",
]
