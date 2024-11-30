"""Factory functions for creating server instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling import config_resources
from llmling.config.manager import ConfigManager
from llmling.config.runtime import RuntimeConfig
from llmling.core.log import get_logger, setup_logging
from llmling.server.server import LLMLingServer


if TYPE_CHECKING:
    import os


logger = get_logger(__name__)


def create_server(
    config_path: str | os.PathLike[str] | None = None,
    *,
    name: str = "llmling-server",
) -> LLMLingServer:
    """Create a fully configured server instance.

    Args:
        config_path: Path to config file (defaults to test config)
        name: Server name

    Returns:
        Configured server instance

    Raises:
        ConfigError: If configuration loading or validation fails
    """
    # Load and validate config
    path = config_path or config_resources.TEST_CONFIG
    manager = ConfigManager.load(path)
    setup_logging(level=manager.config.global_settings.log_level)

    # Log any validation warnings
    if warnings := manager.validate():
        logger.warning("Configuration warnings:\n%s", "\n".join(warnings))

    # Create runtime config
    runtime = RuntimeConfig.from_config(manager.config)

    return LLMLingServer(runtime, name=name)
