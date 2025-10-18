"""Configuration management and validation."""

from __future__ import annotations

import importlib
import re
from typing import TYPE_CHECKING, Self

from llmling.config.models import Config
from llmling.core import exceptions
from llmling.core.log import get_logger, setup_logging


if TYPE_CHECKING:
    from upath.types import JoinablePathLike


logger = get_logger(__name__)
REQ_PATTERN = re.compile(
    r"^([a-zA-Z0-9][a-zA-Z0-9._-]*)(>=|<=|==|!=|>|<|~=)?([0-9a-zA-Z._-]+)?$"
)


class ConfigManager:
    """Manages configuration state and lifecycle.

    Handles loading, storing and validating configurations.
    Foundation for future features like watching and overlays.
    """

    def __init__(self, config: Config | None = None) -> None:
        """Initialize config manager.

        Args:
            config: Optional initial configuration (backward compatibility)
        """
        super().__init__()
        self._configs: list[Config] = []
        if config is not None:
            self._configs.append(config)

    def save(
        self,
        path: JoinablePathLike,
        *,
        validate: bool = True,
    ) -> None:
        """Save configuration to file.

        Args:
            path: Path to save to
            validate: Whether to validate before saving

        Raises:
            ConfigError: If validation fails
        """
        if validate:
            self.validate_or_raise()
        self.config.save(path)

    def validate_or_raise(self) -> None:
        """Run validations and raise on warnings.

        Raises:
            ConfigError: If any validation warnings are found
        """
        if warnings := self.validate():
            msg = "Configuration validation failed:\n" + "\n".join(warnings)
            raise exceptions.ConfigError(msg)

    @property
    def config(self) -> Config:
        """Get first loaded configuration (backward compatibility).

        Returns:
            The first loaded configuration

        Raises:
            ValueError: If no configurations are loaded
        """
        if not self._configs:
            msg = "No configurations loaded"
            raise ValueError(msg)
        return self._configs[0]

    @classmethod
    def load(
        cls,
        path: JoinablePathLike,
        *,
        validate: bool = True,
        strict: bool = False,
    ) -> ConfigManager:
        """Load configuration from file (backward compatibility).

        This classmethod maintains the existing pattern for backward compatibility.
        For new code, prefer using the instance method add_config() instead.

        Args:
            path: Path to configuration file
            validate: Whether to validate the configuration
            strict: Whether to raise on validation warnings

        Returns:
            Configured manager instance

        Raises:
            ConfigError: If loading fails or validation fails with strict=True
        """
        manager = cls()
        manager.add_config(path, validate=validate, strict=strict)
        return manager

    def add_config(
        self,
        path: JoinablePathLike,
        *,
        validate: bool = True,
        strict: bool = False,
    ) -> Config:
        """Add a new configuration.

        Args:
            path: Path to configuration file
            validate: Whether to validate the configuration
            strict: Whether to raise on validation warnings

        Returns:
            Loaded configuration

        Raises:
            ConfigError: If loading fails or validation fails with strict=True
        """
        config = Config.from_file(path)
        # First add to list so validate() works
        self._configs.append(config)

        if validate:
            try:
                if warnings := self.validate():
                    if strict:
                        msg = "Config validation failed:\n" + "\n".join(warnings)
                        raise exceptions.ConfigError(msg)  # noqa: TRY301
                    logger.warning("Config warnings:\n%s", "\n".join(warnings))
            except Exception:
                # Remove config on validation failure
                self._configs.remove(config)
                raise

        setup_logging(level=config.global_settings.log_level)
        return config

    def validate(self) -> list[str]:
        """Validate configuration.

        Performs various validation checks on the configuration including:
        - Resource reference validation
        - Processor configuration validation
        - Tool configuration validation

        Returns:
            List of validation warnings
        """
        config = self.config
        warnings: list[str] = []
        warnings.extend(self._validate_requirements(config))
        warnings.extend(self._validate_resources(config))
        warnings.extend(self._validate_processors(config))
        warnings.extend(self._validate_tools(config))
        return warnings

    def _validate_requirements(self, config: Config) -> list[str]:
        """Validate requirement specifications."""
        from upath import UPath

        warnings = [
            f"Invalid requirement format: {req}"
            for req in config.global_settings.requirements
            if not REQ_PATTERN.match(req)
        ]

        # Validate pip index URL if specified
        if (
            index_url := config.global_settings.pip_index_url
        ) and not index_url.startswith(("http://", "https://")):
            warnings.append(f"Invalid pip index URL: {index_url}")

        # Validate extra paths exist
        for path in config.global_settings.extra_paths:
            try:
                path_obj = UPath(path)
                if not path_obj.exists():
                    warnings.append(f"Extra path does not exist: {path}")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Invalid extra path {path}: {exc}")

        return warnings

    def _validate_resources(self, config: Config) -> list[str]:
        """Validate resource configuration."""
        warnings: list[str] = []

        # Check resource group references
        warnings.extend(
            f"Resource {resource} in group {group} not found"
            for group, resources in config.resource_groups.items()
            for resource in resources
            if resource not in config.resources
        )

        # Check processor references in resources
        warnings.extend(
            f"Processor {proc.name} in resource {name} not found"
            for name, resource in config.resources.items()
            for proc in resource.processors
            if proc.name not in config.context_processors
        )

        # Resource-specific validation
        for resource in config.resources.values():
            warnings.extend(resource.validate_resource())

        return warnings

    def _validate_processors(self, config: Config) -> list[str]:
        """Validate processor configuration."""
        warnings = []
        for name, processor in config.context_processors.items():
            if not processor.import_path:
                warnings.append(f"Processor {name} missing import_path")
                continue

            # Try to import the module
            try:
                importlib.import_module(processor.import_path.split(".")[0])
            except ImportError:
                path = processor.import_path
                msg = f"Cannot import module for processor {name}: {path}"
                warnings.append(msg)

        return warnings

    def _validate_tools(self, config: Config) -> list[str]:
        """Validate tool configuration."""
        warnings = []

        for name, tool in config.tools.items():
            if not tool.import_path:
                warnings.append(f"Tool {name} missing import_path")
                # Check for duplicate tool names
            warnings.extend(
                f"Tool {name} defined both explicitly and in toolset"
                for toolset_tool in config.toolsets
                if toolset_tool == name
            )

        return warnings

    async def __aenter__(self) -> Self:
        """Enter async context."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Exit async context."""
        self._configs.clear()
