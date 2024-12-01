"""Factory functions for creating server instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

import depkit

from llmling import config_resources
from llmling.config.manager import ConfigManager
from llmling.config.runtime import RuntimeConfig
from llmling.core.log import get_logger
from llmling.server.server import LLMLingServer


if TYPE_CHECKING:
    import os


logger = get_logger(__name__)


def create_server(
    config_path: str | os.PathLike[str] | None = None,
    *,
    name: str = "llmling-server",
    install_requirements: bool = True,
) -> LLMLingServer:
    """Create a fully configured server instance.

    Args:
        config_path: Path to config file (defaults to test config)
        name: Server name
        install_requirements: Whether to install missing requirements

    Returns:
        Configured server instance

    Raises:
        ConfigError: If configuration loading or validation fails
    """
    # Load and validate config
    path = config_path or config_resources.TEST_CONFIG
    manager = ConfigManager.load(path)

    # Log any validation warnings
    if warnings := manager.validate():
        logger.warning("Configuration warnings:\n%s", "\n".join(warnings))

    # Handle dependencies if requested
    with depkit.DependencyManager(
        prefer_uv=manager.config.global_settings.prefer_uv,
        requirements=manager.config.global_settings.requirements,
        extra_paths=manager.config.global_settings.extra_paths,
        pip_index_url=manager.config.global_settings.pip_index_url,
        scripts=manager.config.global_settings.scripts,
    ):
        # Create runtime config
        runtime = RuntimeConfig.from_config(manager.config)
        return LLMLingServer(runtime, name=name)


if __name__ == "__main__":
    import depkit

    manager = depkit.DependencyManager()
    print(manager)
