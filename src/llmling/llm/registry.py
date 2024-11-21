"""LLM provider registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling.core import exceptions
from llmling.core.baseregistry import BaseRegistry
from llmling.core.log import get_logger
from llmling.llm.base import LLMProvider
from llmling.llm.providers.litellm import LiteLLMProvider


if TYPE_CHECKING:
    from llmling.llm.base import LLMConfig


logger = get_logger(__name__)


class ProviderRegistry(BaseRegistry[type[LLMProvider], str]):
    """Registry for LLM providers."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        super().__init__()
        self._implementations: dict[str, type[LLMProvider]] = {
            "litellm": LiteLLMProvider,
        }

    @property
    def _error_class(self) -> type[exceptions.LLMError]:
        return exceptions.LLMError

    def _validate_item(self, item: Any) -> type[LLMProvider]:
        """Validate and possibly transform item before registration."""
        if isinstance(item, type) and issubclass(item, LLMProvider):
            return item
        msg = f"Invalid provider type: {type(item)}"
        raise exceptions.LLMError(msg)

    # Backward compatibility methods
    def register_provider(self, name: str, implementation: str) -> None:
        """Register a provider configuration with an implementation."""
        if name in self._items:
            if implementation not in self._implementations:
                msg = f"Implementation {implementation} not found"
                raise exceptions.LLMError(msg)
            if self._items[name] != self._implementations[implementation]:
                msg = (
                    f"Provider {name} already registered with a different implementation"
                )
                raise exceptions.LLMError(msg)
            return  # Already registered with the same implementation
        if implementation not in self._implementations:
            msg = f"Implementation {implementation} not found"
            raise exceptions.LLMError(msg)
        super().register(name, self._implementations[implementation])

    def create_provider(self, name: str, config: LLMConfig) -> LLMProvider:
        """Create a provider instance."""
        provider_class = self.get(name)
        return provider_class(config)

    def reset(self) -> None:
        """Reset the registry to its initial state."""
        super().reset()
        self._implementations = {
            "litellm": LiteLLMProvider,
        }


# Global registry instance
default_registry = ProviderRegistry()
