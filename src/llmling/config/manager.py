"""Configuration management utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from upath import UPath
import yamling

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.extensions.loaders import ToolsetLoader
from llmling.tools.base import LLMCallableTool
from llmling.utils import importing


if TYPE_CHECKING:
    import os

    from llmling.config.models import Config, Resource, ToolConfig


logger = get_logger(__name__)


class ConfigManager:
    """Configuration management system."""

    def __init__(self, config: Config) -> None:
        """Initialize with configuration.

        Args:
            config: Application configuration
        """
        self.config = config

    def register_resource(
        self,
        name: str,
        resource: Resource,
        *,
        replace: bool = False,
    ) -> None:
        """Register a new resource."""
        if name in self.config.resources and not replace:
            msg = f"Resource already exists: {name}"
            raise exceptions.ConfigError(msg)
        self.config.resources[name] = resource

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> ConfigManager:
        """Load configuration from file.

        Args:
            path: Path to configuration file

        Returns:
            Configuration manager instance

        Raises:
            ConfigError: If loading fails
        """
        from llmling.config.loading import load_config

        config = load_config(path)
        return cls(config)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save configuration to file.

        Args:
            path: Path to save configuration

        Raises:
            ConfigError: If saving fails
        """
        try:
            content = self.config.model_dump(exclude_none=True)
            string = yamling.dump_yaml(content)
            _ = UPath(path).write_text(string)

        except Exception as exc:
            msg = f"Failed to save configuration to {path}"
            raise exceptions.ConfigError(msg) from exc

    def validate_references(self) -> list[str]:
        """Validate all references in configuration.

        Returns:
            List of validation warnings
        """
        # Check resource references
        return [
            f"Resource {resource} in group {group} not found"
            for group, resources in self.config.resource_groups.items()
            for resource in resources
            if resource not in self.config.resources
        ]

    def _create_tool(self, tool_config: ToolConfig) -> LLMCallableTool:
        """Create tool instance from config.

        Args:
            tool_config: Tool configuration

        Returns:
            Configured tool instance

        Raises:
            ConfigError: If tool creation fails
        """
        try:
            callable_obj = importing.import_callable(tool_config.import_path)
            return LLMCallableTool.from_callable(
                callable_obj,
                name_override=tool_config.name,
                description_override=tool_config.description,
            )
        except Exception as exc:
            msg = f"Failed to create tool from {tool_config.import_path}"
            raise exceptions.ConfigError(msg) from exc

    def get_tools(self) -> dict[str, LLMCallableTool]:
        """Get all tools from config and toolsets."""
        tools = {}

        # Load explicitly configured tools
        for name, tool_config in self.config.tools.items():
            try:
                tools[name] = self._create_tool(tool_config)
            except Exception:
                logger.exception("Failed to create tool %s", name)

        # Load tools from toolsets
        if self.config.toolsets:
            loader = ToolsetLoader()
            toolset_tools = loader.load_items(self.config.toolsets)

            # Handle potential name conflicts
            for name, tool in toolset_tools.items():
                if name in tools:
                    logger.warning(
                        "Tool %s from toolset overlaps with configured tool", name
                    )
                    continue
                tools[name] = tool

        return tools
