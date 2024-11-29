"""Configuration management and validation.

This module provides the ConfigManager class which handles all static configuration
operations including loading, validation, and saving.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import logfire
from upath import UPath
import yamling

from llmling.config.models import PromptConfig
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.prompts.models import Prompt


if TYPE_CHECKING:
    import os

    from llmling.config.models import Config


logger = get_logger(__name__)


class ConfigManager:
    """Manages and validates static configuration.

    This class handles all operations on the static configuration including
    validation, saving, and loading. It ensures configuration integrity
    before it gets transformed into runtime state.
    """

    def __init__(self, config: Config) -> None:
        """Initialize with configuration.

        Args:
            config: Configuration to manage
        """
        self._config = config

    @property
    def config(self) -> Config:
        """Get the managed configuration."""
        return self._config

    @config.setter
    def config(self, value: Config) -> None:
        self._config = value

    def save(
        self,
        path: str | os.PathLike[str],
        *,
        validate: bool = True,
    ) -> None:
        """Save configuration to file.

        Args:
            path: Path to save to
            validate: Whether to validate before saving

        Raises:
            ConfigError: If validation or saving fails
        """
        try:
            if validate:
                self.validate_or_raise()

            content = self.config.model_dump(exclude_none=True)
            string = yamling.dump_yaml(content)
            UPath(path).write_text(string)
            logger.info("Configuration saved to %s", path)

        except Exception as exc:
            msg = f"Failed to save configuration to {path}"
            raise exceptions.ConfigError(msg) from exc

    @logfire.instrument("Validating configuration")
    def validate(self) -> list[str]:
        """Validate configuration.

        Performs various validation checks on the configuration including:
        - Resource reference validation
        - Processor configuration validation
        - Tool configuration validation

        Returns:
            List of validation warnings
        """
        warnings: list[str] = []
        warnings.extend(self._validate_resources())
        warnings.extend(self._validate_processors())
        warnings.extend(self._validate_tools())
        return warnings

    def validate_or_raise(self) -> None:
        """Run validations and raise on warnings.

        Raises:
            ConfigError: If any validation warnings are found
        """
        if warnings := self.validate():
            msg = "Configuration validation failed:\n" + "\n".join(warnings)
            raise exceptions.ConfigError(msg)

    def _validate_prompts(self) -> list[str]:
        """Validate prompt configuration."""
        warnings = []
        for name, prompt_config in self.config.prompts.items():
            match prompt_config:
                case PromptConfig():
                    if not prompt_config.import_path:
                        warnings.append(f"Prompt {name} missing import_path")
                    else:
                        # Try to import the module
                        try:
                            name = prompt_config.import_path.split(".")[0]
                            importlib.import_module(name)
                        except ImportError:
                            warnings.append(
                                f"Cannot import module for prompt {name}: "
                                f"{prompt_config.import_path}"
                            )
                case Prompt():
                    if not prompt_config.messages:
                        warnings.append(f"Prompt {name} has no messages")
        return warnings

    def _validate_resources(self) -> list[str]:
        """Validate resource configuration.

        Returns:
            List of validation warnings
        """
        warnings: list[str] = []

        # Check resource group references
        warnings.extend(
            f"Resource {resource} in group {group} not found"
            for group, resources in self.config.resource_groups.items()
            for resource in resources
            if resource not in self.config.resources
        )

        # Check processor references in resources
        warnings.extend(
            f"Processor {proc.name} in resource {name} not found"
            for name, resource in self.config.resources.items()
            for proc in resource.processors
            if proc.name not in self.config.context_processors
        )

        # Check resource paths exist for local resources
        for resource in self.config.resources.values():
            if hasattr(resource, "path"):
                path = UPath(resource.path)
                prefixes = ("http://", "https://")
                if not path.exists() and not path.as_uri().startswith(prefixes):
                    warnings.append(f"Resource path not found: {path}")

        return warnings

    def _validate_processors(self) -> list[str]:
        """Validate processor configuration."""
        warnings = []
        for name, processor in self.config.context_processors.items():
            match processor.type:
                case "function":
                    if not processor.import_path:
                        warnings.append(f"Processor {name} missing import_path")
                    else:
                        # Try to import the module
                        try:
                            importlib.import_module(processor.import_path.split(".")[0])
                        except ImportError:
                            path = processor.import_path
                            warnings.append(
                                f"Cannot import module for processor {name}: {path}"
                            )
                case "template":
                    if not processor.template:
                        warnings.append(f"Processor {name} missing template")
        return warnings

    def _validate_tools(self) -> list[str]:
        """Validate tool configuration.

        Returns:
            List of validation warnings
        """
        warnings = []

        for name, tool in self.config.tools.items():
            if not tool.import_path:
                warnings.append(f"Tool {name} missing import_path")
                # Check for duplicate tool names
            warnings.extend(
                f"Tool {name} defined both explicitly and in toolset"
                for toolset_tool in self.config.toolsets
                if toolset_tool == name
            )

        return warnings

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> ConfigManager:
        """Load configuration from file.

        Args:
            path: Path to configuration file

        Returns:
            ConfigManager instance

        Raises:
            ConfigError: If loading fails
        """
        from llmling.config.loading import load_config

        try:
            config = load_config(path)
            return cls(config)
        except Exception as exc:
            msg = f"Failed to load configuration from {path}"
            raise exceptions.ConfigError(msg) from exc
