"""LLM provider registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling.core import exceptions
from llmling.core.baseregistry import BaseRegistry
from llmling.core.log import get_logger
from llmling.llm.base import LLMProvider
from llmling.llm.providers.litellmprovider import LiteLLMProvider


if TYPE_CHECKING:
    from llmling.llm.base import LLMConfig

logger = get_logger(__name__)


class ProviderFactory:
    """Factory to create configured provider instances."""

    def __init__(self, provider_class: type[LLMProvider]) -> None:
        """Initialize with provider class."""
        self.provider_class = provider_class

    def create(self, config: LLMConfig) -> LLMProvider:
        """Create new provider instance with config."""
        return self.provider_class(config)


class ProviderRegistry(BaseRegistry[str, ProviderFactory]):
    """Registry for LLM provider factories."""

    def __init__(self) -> None:
        """Initialize registry."""
        super().__init__()
        # Register default implementation
        self.register("litellm", ProviderFactory(LiteLLMProvider))

    @property
    def _error_class(self) -> type[exceptions.LLMError]:
        return exceptions.LLMError

    def _validate_item(self, item: Any) -> ProviderFactory:
        """Validate and create factory from input."""
        try:
            match item:
                # Handle direct provider class
                case type() if issubclass(item, LLMProvider):
                    return ProviderFactory(item)
                # Handle factory instance
                case ProviderFactory():
                    return item
                # Handle string reference to litellm
                case str():
                    return ProviderFactory(LiteLLMProvider)
                case _:
                    msg = f"Invalid provider type: {type(item)}"
                    raise exceptions.LLMError(msg)
        except TypeError as exc:
            msg = f"Invalid provider: {exc}"
            raise exceptions.LLMError(msg) from exc

    def create_provider(self, name: str, config: LLMConfig) -> LLMProvider:
        """Create a configured provider instance."""
        try:
            factory = self.get(name)
            return factory.create(config)
        except KeyError as exc:
            msg = f"Provider not found: {name}"
            raise exceptions.LLMError(msg) from exc


# Global registry instance
default_registry = ProviderRegistry()
