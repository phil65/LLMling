"""LLM provider registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.llm.base import LLMConfig  # noqa: TCH001
from llmling.llm.providers.litellm import LiteLLMProvider


if TYPE_CHECKING:
    from llmling.llm.base import LLMProvider


logger = get_logger(__name__)


class ProviderRegistry:
    """Registry for LLM providers."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._providers: dict[str, type[LLMProvider]] = {
            "litellm": LiteLLMProvider,
        }

    def register_provider(
        self,
        name: str,
        provider_class: type[LLMProvider],
    ) -> None:
        """Register a new provider.

        Args:
            name: Provider name
            provider_class: Provider class to register

        Raises:
            LLMError: If provider already registered
        """
        if name in self._providers:
            msg = f"Provider {name} already registered"
            raise exceptions.LLMError(msg)

        logger.info("Registering LLM provider: %s", name)
        self._providers[name] = provider_class

    def create_provider(
        self,
        name: str,
        config: LLMConfig,
    ) -> LLMProvider:
        """Create a provider instance.

        Args:
            name: Provider name
            config: Provider configuration

        Returns:
            Provider instance

        Raises:
            LLMError: If provider not found
        """
        try:
            provider_class = self._providers[name]
            return provider_class(config)
        except KeyError as exc:
            msg = f"Provider not found: {name}"
            raise exceptions.LLMError(msg) from exc

    def get_provider_class(self, name: str) -> type[LLMProvider]:
        """Get a provider class by name.

        Args:
            name: Provider name

        Returns:
            Provider class

        Raises:
            LLMError: If provider not found
        """
        try:
            return self._providers[name]
        except KeyError as exc:
            msg = f"Provider not found: {name}"
            raise exceptions.LLMError(msg) from exc


# Global registry instance
default_registry = ProviderRegistry()
