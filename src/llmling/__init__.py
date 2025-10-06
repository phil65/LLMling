"""LLMling: main package.

LLM simplified.
"""

from __future__ import annotations

from importlib.metadata import version

__version__ = version("llmling")
__title__ = "LLMling"

__author__ = "Philipp Temminghoff"
__author_email__ = "philipptemminghoff@googlemail.com"
__copyright__ = "Copyright (c) 2024 Philipp Temminghoff"
__license__ = "MIT"
__url__ = "https://github.com/phil65/llmling"

from upathtools import register_http_filesystems

register_http_filesystems()

from llmling.resources import (
    ResourceLoader,
    LoadedResource,
    default_registry as resource_registry,
)
from llmling.config.runtime import RuntimeConfig
from llmling.core.exceptions import (
    LLMLingError,
    ConfigError,
    ResourceError,
    LoaderError,
    ProcessorError,
    LLMError,
)
from llmling.processors.registry import ProcessorRegistry
from llmling.tools import LLMCallableTool, ToolError
from llmling.prompts import (
    PromptMessage,
    PromptParameter,
    StaticPrompt,
    DynamicPrompt,
    BasePrompt,
)
from llmling.config.models import (
    ConfigModel,
    GlobalSettings,
    LLMCapabilitiesConfig,
    Config,
)
from llmling.config.store import ConfigStore
from llmling.core.baseregistry import BaseRegistry


__all__ = [
    "BasePrompt",
    "BaseRegistry",
    "Config",
    "ConfigError",
    "ConfigModel",
    "ConfigStore",
    "DynamicPrompt",
    "GlobalSettings",
    "LLMCallableTool",
    "LLMCapabilitiesConfig",
    "LLMError",
    "LLMLingError",
    "LoadedResource",
    "LoaderError",
    "ProcessorError",
    "ProcessorRegistry",
    "PromptMessage",
    "PromptParameter",
    "ResourceError",
    "ResourceLoader",
    "RuntimeConfig",
    "StaticPrompt",
    "ToolError",
    "__version__",
    "resource_registry",
]
