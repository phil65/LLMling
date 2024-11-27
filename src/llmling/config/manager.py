"""Configuration management utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from upath import UPath
import yamling

from llmling.core import exceptions
from llmling.core.log import get_logger


if TYPE_CHECKING:
    import os

    from llmling.config.models import Config, Resource


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
