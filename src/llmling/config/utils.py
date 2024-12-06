"""Utility functions for the config package."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from llmling.config.manager import ConfigManager
from llmling.config.models import (
    Config,
    CustomToolsetConfig,
    EntryPointToolsetConfig,
    OpenAPIToolsetConfig,
)
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.tools.entry_points import EntryPointTools
from llmling.tools.openapi import OpenAPITools
from llmling.utils import importing


if TYPE_CHECKING:
    from llmling.config.runtime import RuntimeConfig
    from llmling.tools.toolsets import ToolSet

logger = get_logger(__name__)


def toolset_config_to_toolset(config) -> ToolSet:
    match config:
        case OpenAPIToolsetConfig():
            return OpenAPITools(
                spec=config.spec,
                base_url=config.base_url or "",
            )
        case EntryPointToolsetConfig():
            return EntryPointTools(config.module)
        case CustomToolsetConfig():
            toolset_class = importing.import_class(config.import_path)
            return toolset_class()
        case _:
            msg = f"Unknown toolset type: {type(config)}"
            raise ValueError(msg)


def prepare_runtime(
    runtime_cls: type[RuntimeConfig],
    source: str | os.PathLike[str] | Config,
    *,
    validate: bool = True,
    strict: bool = False,
) -> RuntimeConfig:
    """Prepare runtime configuration from source.

    Args:
        runtime_cls: RuntimeConfig class to instantiate
        source: Path to configuration file or Config object
        validate: Whether to validate config
        strict: Whether to raise on validation warnings

    Returns:
        Initialized runtime configuration

    Raises:
        TypeError: If source type is invalid
        ConfigError: If validation fails in strict mode
    """
    match source:
        case str() | os.PathLike():
            manager = ConfigManager.load(source, validate=validate, strict=strict)
            config = manager.config
        case Config():
            config = source
            if validate:
                manager = ConfigManager(config)
                if warnings := manager.validate():
                    if strict:
                        msg = "Config validation failed:\n" + "\n".join(warnings)
                        raise exceptions.ConfigError(msg)
                    logger.warning("Config warnings:\n%s", "\n".join(warnings))
        case _:
            msg = f"Invalid source type: {type(source)}"
            raise TypeError(msg)

    return runtime_cls.from_config(config)
