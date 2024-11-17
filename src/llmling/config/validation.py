"""Configuration validation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import logfire

from llmling.core import exceptions
from llmling.core.log import get_logger


if TYPE_CHECKING:
    from llmling.config.models import Config


logger = get_logger(__name__)


class ConfigValidator:
    """Configuration validator."""

    def __init__(self, config: Config) -> None:
        """Initialize validator.

        Args:
            config: Configuration to validate
        """
        self.config = config

    def validate_all(self) -> list[str]:
        """Run all validations.

        Returns:
            List of validation warnings
        """
        warnings = []
        warnings.extend(self._validate_providers())
        warnings.extend(self._validate_contexts())
        warnings.extend(self._validate_processors())
        warnings.extend(self._validate_templates())
        return warnings

    def _validate_providers(self) -> list[str]:
        """Validate provider configuration.

        Returns:
            List of validation warnings
        """
        warnings = []

        # Check provider groups
        for group, providers in self.config.provider_groups.items():
            for provider in providers:
                if provider not in self.config.llm_providers:
                    warnings.append(
                        f"Provider {provider} in group {group} not found",
                    )

        # Check provider models
        for name, provider in self.config.llm_providers.items():
            if "/" not in provider.model:
                warnings.append(
                    f"Provider {name} model should be in format 'provider/model'",
                )

        return warnings

    def _validate_contexts(self) -> list[str]:
        """Validate context configuration.

        Returns:
            List of validation warnings
        """
        warnings: list[str] = []

        warnings.extend(
            f"Context {context} in group {group} not found"
            for group, contexts in self.config.context_groups.items()
            for context in contexts
            if context not in self.config.contexts
        )

        warnings.extend(
            f"Processor {processor} in context {name} not found"
            for name, context in self.config.contexts.items()
            for processor in context.processors
            if processor.name not in self.config.context_processors
        )

        return warnings

    def _validate_processors(self) -> list[str]:
        """Validate processor configuration.

        Returns:
            List of validation warnings
        """
        warnings = []

        for name, processor in self.config.context_processors.items():
            if processor.type == "function" and not processor.import_path:
                warnings.append(
                    f"Processor {name} missing import_path for type 'function'",
                )
            elif processor.type == "template" and not processor.template:
                warnings.append(
                    f"Processor {name} missing template for type 'template'",
                )

        return warnings

    def _validate_templates(self) -> list[str]:
        """Validate template configuration.

        Returns:
            List of validation warnings
        """
        warnings = []

        for name, template in self.config.task_templates.items():
            # Validate provider reference
            if (
                template.provider not in self.config.llm_providers
                and template.provider not in self.config.provider_groups
            ):
                warnings.append(
                    f"Provider {template.provider} in template {name} not found",
                )

            # Validate context reference
            if (
                template.context not in self.config.contexts
                and template.context not in self.config.context_groups
            ):
                warnings.append(
                    f"Context {template.context} in template {name} not found",
                )

        return warnings

    @logfire.instrument("Validating configs")
    def validate_or_raise(self) -> None:
        """Run all validations and raise on warnings.

        Raises:
            ConfigError: If any validation warnings are found
        """
        warnings = self.validate_all()
        if warnings:
            msg = "Configuration validation failed:\n" + "\n".join(warnings)
            raise exceptions.ConfigError(msg)

    def validate_references(self) -> list[str]:
        """Validate all references in configuration."""
        warnings = []

        # Validate provider references
        for group_name, providers in self.config.provider_groups.items():
            for provider_name in providers:
                if provider_name not in self.config.llm_providers:
                    warnings.append(
                        f"Provider {provider_name} in group {group_name} not found"
                    )

        # Validate context references
        for template_name, template in self.config.task_templates.items():
            provider = template.provider
            if (
                provider not in self.config.llm_providers
                and provider not in self.config.provider_groups
            ):
                warnings.append(
                    f"Provider {provider} in template {template_name} not found"
                )

        return warnings
