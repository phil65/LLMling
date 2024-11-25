"""Configuration loading utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import logfire
from upath import UPath
import yamling

from llmling.config.models import Config
from llmling.config.validation import ConfigValidator
from llmling.core import exceptions
from llmling.core.log import get_logger


if TYPE_CHECKING:
    import os


logger = get_logger(__name__)


@logfire.instrument("Loading configuration from {path}")
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
    logger.debug("Loading configuration from %s", path)  # Use debug level

    try:
        content = yamling.load_yaml_file(path)
    except Exception as exc:
        msg = f"Failed to load YAML from {path!r}"
        raise exceptions.ConfigError(msg) from exc
    # Validate basic structure
    if not isinstance(content, dict):
        msg = "Configuration must be a dictionary"
        raise exceptions.ConfigError(msg)
    try:
        # Convert to model
        config = Config.model_validate(content)

        # Validate references if requested
        if validate:
            logger.debug("Validating configuration")  # Use debug level
            validator = ConfigValidator(config)
            validator.validate_or_raise()
    except Exception as exc:
        msg = f"Failed to load configuration from {path}"
        raise exceptions.ConfigError(msg) from exc
    else:
        msg = "Loaded configuration: version=%s, contexts=%d"
        logger.info(msg, config.version, len(config.contexts))
        return config


def save_config(
    config: Config,
    path: str | os.PathLike[str],
    *,
    validate: bool = True,
) -> None:
    """Save configuration to file.

    Args:
        config: Configuration to save
        path: Path to save to
        validate: Whether to validate before saving

    Raises:
        ConfigError: If validation or saving fails
    """
    logger.debug("Saving configuration to %s", path)

    try:
        if validate:
            validator = ConfigValidator(config)
            validator.validate_or_raise()

        # Convert to dictionary
        content = config.model_dump(exclude_none=True, by_alias=True)

        # Save to file
        string = yamling.dump_yaml(content)
        UPath(path).write_text(string)
        logger.info("Configuration saved successfully")

    except Exception as exc:
        msg = f"Failed to save configuration to {path}"
        raise exceptions.ConfigError(msg) from exc


if __name__ == "__main__":
    # Example usage
    import sys

    try:
        config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
        config = load_config(config_path)
        print(f"Loaded configuration version: {config.version}")
        print(f"Number of contexts: {len(config.contexts)}")
    except exceptions.ConfigError as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)
