"""LLM provider registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.llm.base import LLMConfig
from llmling.llm.providers.litellm import LiteLLMProvider


if TYPE_CHECKING:
    from llmling.llm.base import LLMProvider


logger = get_logger(__name__)


class ProviderRegistry:
    """Registry for LLM providers."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        # Map of provider implementations (like 'litellm' -> LiteLLMProvider)
        self._implementations: dict[str, type[LLMProvider]] = {
            "litellm": LiteLLMProvider,
        }
        # Map of configured providers to their implementations
        self._providers: dict[str, str] = {}

    def register_provider(
        self,
        name: str,
        implementation: str,
    ) -> None:
        """Register a provider configuration with an implementation.

        Args:
            name: Name of the provider configuration (e.g., 'gpt4-turbo')
            implementation: Name of the implementation to use (e.g., 'litellm')

        Raises:
            LLMError: If provider already registered or implementation not found
        """
        if name in self._providers:
            msg = f"Provider {name} already registered"
            raise exceptions.LLMError(msg)

        if implementation not in self._implementations:
            msg = f"Implementation {implementation} not found"
            raise exceptions.LLMError(msg)

        logger.info("Registering LLM provider %s using %s", name, implementation)
        self._providers[name] = implementation

    def register_implementation(
        self,
        name: str,
        provider_class: type[LLMProvider],
    ) -> None:
        """Register a new provider implementation.

        Args:
            name: Implementation name (e.g., 'litellm')
            provider_class: Provider implementation class

        Raises:
            LLMError: If implementation already registered
        """
        if name in self._implementations:
            msg = f"Implementation {name} already registered"
            raise exceptions.LLMError(msg)

        logger.info("Registering LLM implementation: %s", name)
        self._implementations[name] = provider_class

    def create_provider(
        self,
        name: str,
        config: LLMConfig,
    ) -> LLMProvider:
        """Create a provider instance.

        Args:
            name: Provider name from configuration
            config: Provider configuration

        Returns:
            Provider instance

        Raises:
            LLMError: If provider not found
        """
        try:
            # Get the implementation for this provider
            implementation = self._providers.get(name)
            if not implementation:
                msg = f"Provider not found: {name}"
                raise exceptions.LLMError(msg)

            # Get the implementation class
            provider_class = self._implementations[implementation]

            # Create and return provider instance
            return provider_class(config)

        except KeyError as exc:
            msg = f"Provider or implementation not found: {name}"
            raise exceptions.LLMError(msg) from exc

    def get_implementation(self, name: str) -> type[LLMProvider]:
        """Get a provider implementation by name.

        Args:
            name: Implementation name

        Returns:
            Provider implementation class

        Raises:
            LLMError: If implementation not found
        """
        try:
            return self._implementations[name]
        except KeyError as exc:
            msg = f"Implementation not found: {name}"
            raise exceptions.LLMError(msg) from exc


# Global registry instance
default_registry = ProviderRegistry()
