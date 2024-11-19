"""LLM provider registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.llm.providers.litellm import LiteLLMProvider


if TYPE_CHECKING:
    from llmling.llm.base import LLMConfig, LLMProvider


logger = get_logger(__name__)


class ProviderRegistry:
    """Registry for LLM providers."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._implementations: dict[str, type[LLMProvider]] = {
            "litellm": LiteLLMProvider,
        }
        self._providers: dict[str, str] = {}

    def register_provider(
        self,
        name: str,
        implementation: str,
    ) -> None:
        """Register a provider configuration with an implementation.

        If the provider is already registered with the same implementation,
        this is a no-op. If it's registered with a different implementation,
        an error is raised.
        """
        if name in self._providers:
            if self._providers[name] != implementation:
                msg = f"Provider {name} already registered with different implementation"
                raise exceptions.LLMError(msg)
            return  # Already registered with same implementation

        if implementation not in self._implementations:
            msg = f"Implementation {implementation} not found"
            raise exceptions.LLMError(msg)

        logger.debug("Registering LLM provider %s using %s", name, implementation)
        self._providers[name] = implementation

    def create_provider(
        self,
        name: str,
        config: LLMConfig,
    ) -> LLMProvider:
        """Create a provider instance."""
        try:
            if not (implementation := self._providers.get(name)):
                msg = f"Provider not found: {name}"
                raise exceptions.LLMError(msg)

            provider_class = self._implementations[implementation]
            return provider_class(config)

        except KeyError as exc:
            msg = f"Provider or implementation not found: {name}"
            raise exceptions.LLMError(msg) from exc

    def reset(self) -> None:
        """Reset the registry to its initial state."""
        self._providers.clear()


# Global registry instance
default_registry = ProviderRegistry()
