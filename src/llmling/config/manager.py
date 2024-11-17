"""Configuration management utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from upath import UPath
import yamling

from llmling.core import exceptions
from llmling.core.log import get_logger


if TYPE_CHECKING:
    import os

    from llmling.config.models import Config


logger = get_logger(__name__)


class ConfigManager:
    """Configuration management system."""

    def __init__(self, config: Config) -> None:
        """Initialize with configuration.

        Args:
            config: Application configuration
        """
        self.config = config

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
            UPath(path).write_text(string)

        except Exception as exc:
            msg = f"Failed to save configuration to {path}"
            raise exceptions.ConfigError(msg) from exc

    def get_effective_settings(
        self,
        template_name: str,
    ) -> dict[str, Any]:
        """Get effective settings for a template.

        Args:
            template_name: Template name

        Returns:
            Combined settings from global and template

        Raises:
            ConfigError: If template not found
        """
        try:
            template = self.config.task_templates[template_name]

            # Start with global settings
            settings = self.config.global_settings.model_dump()

            # Add provider settings if direct provider
            if template.provider in self.config.llm_providers:
                provider = self.config.llm_providers[template.provider]
                settings.update(provider.model_dump(exclude_none=True))

            # Add template settings
            if template.settings:
                settings.update(template.settings.model_dump(exclude_none=True))

            return settings

        except KeyError as exc:
            msg = f"Template not found: {template_name}"
            raise exceptions.ConfigError(msg) from exc

    def validate_references(self) -> list[str]:
        """Validate all references in configuration.

        Returns:
            List of validation warnings
        """
        warnings = []

        # Check provider references
        for group, providers in self.config.provider_groups.items():
            for provider in providers:
                if provider not in self.config.llm_providers:
                    warnings.append(
                        f"Provider {provider} in group {group} not found",
                    )

        # Check context references
        for group, contexts in self.config.context_groups.items():
            for context in contexts:
                if context not in self.config.contexts:
                    warnings.append(
                        f"Context {context} in group {group} not found",
                    )

        # Check template references
        for name, template in self.config.task_templates.items():
            # Check provider reference
            if (
                template.provider not in self.config.llm_providers
                and template.provider not in self.config.provider_groups
            ):
                warnings.append(
                    f"Provider {template.provider} in template {name} not found",
                )

            # Check context reference
            if (
                template.context not in self.config.contexts
                and template.context not in self.config.context_groups
            ):
                warnings.append(
                    f"Context {template.context} in template {name} not found",
                )

        return warnings
