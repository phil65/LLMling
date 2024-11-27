"""Configuration file handling.

This module provides the ConfigFile class for managing YAML configuration files.
It handles loading, saving, and path resolution for LLMling configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from upath import UPath
import yamling

from llmling.config.models import Config
from llmling.config.validation import ConfigValidator
from llmling.core import exceptions
from llmling.core.log import get_logger


if TYPE_CHECKING:
    import os


logger = get_logger(__name__)


class ConfigFile:
    """Manages a LLMling configuration file.

    This class handles loading and saving of configuration files, with built-in
    path resolution and validation. Configuration is loaded immediately upon
    instantiation and is always available through the `config` property.

    Args:
        path: Path to configuration file

    Raises:
        ConfigError: If loading or validation fails

    Example:
        >>> try:
        ...     config_file = ConfigFile("config.yml")
        ...     print(f"Loaded {len(config_file.config.resources)} resources")
        ... except exceptions.ConfigError as e:
        ...     print(f"Failed to load config: {e}")
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """Initialize and load configuration."""
        self.path = UPath(path).resolve()

        try:
            logger.debug("Loading configuration from %s", self.path)
            content = yamling.load_yaml_file(self.path)

            if not isinstance(content, dict):
                msg = "Configuration must be a dictionary"
                raise exceptions.ConfigError(msg)  # noqa: TRY301

            self._config = Config.model_validate(content)

            # Resolve paths relative to config location
            if hasattr(self._config, "resolve_paths"):
                self._config.resolve_paths(self.path.parent)

            logger.info(
                "Loaded configuration: version=%s, resources=%d",
                self._config.version,
                len(self._config.resources),
            )

        except Exception as exc:
            msg = f"Failed to load configuration from {self.path}"
            raise exceptions.ConfigError(msg) from exc

    def __repr__(self) -> str:
        """Return string representation of the config file."""
        return f"ConfigFile(path='{self.path}')"

    @property
    def config(self) -> Config:
        """The loaded configuration.

        Returns:
            Current configuration
        """
        return self._config

    def reload(self) -> None:
        """Reload configuration from file.

        This method re-reads the configuration file and updates the current
        configuration. All paths are re-resolved relative to the config file
        location.

        Raises:
            ConfigError: If reloading fails
        """
        self.__init__(self.path)  # type: ignore[misc]

    def save(self, *, validate: bool = True) -> None:
        """Save current configuration to file.

        Args:
            validate: Whether to validate before saving

        Raises:
            ConfigError: If saving or validation fails
        """
        try:
            if validate:
                validator = ConfigValidator(self._config)
                validator.validate_or_raise()

            content = self._config.model_dump(exclude_none=True, by_alias=True)
            string = yamling.dump_yaml(content)
            self.path.write_text(string)
            logger.info("Configuration saved to %s", self.path)

        except Exception as exc:
            msg = f"Failed to save configuration to {self.path}"
            raise exceptions.ConfigError(msg) from exc


def load_config(path: str | os.PathLike[str], *, validate: bool = True) -> Config:
    """Load and validate configuration from YAML file.

    Args:
        path: Path to configuration file
        validate: Whether to validate the configuration

    Returns:
        Loaded configuration

    Raises:
        ConfigError: If loading or validation fails
    """
    try:
        config_file = ConfigFile(path)

        if validate:
            validator = ConfigValidator(config_file.config)
            validator.validate_or_raise()
    except Exception as exc:
        msg = f"Failed to load config from {path}"
        raise exceptions.ConfigError(msg) from exc
    else:
        return config_file.config


if __name__ == "__main__":
    import sys

    try:
        config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
        config_file = ConfigFile(config_path)
        print(f"Loaded config file: {config_file}")
        print(f"Version: {config_file.config.version}")
        print(f"Resources: {len(config_file.config.resources)}")
    except exceptions.ConfigError as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)
