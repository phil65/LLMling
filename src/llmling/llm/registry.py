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
        # Store actual implementations
        self._implementations: dict[str, type[LLMProvider]] = {
            "litellm": LiteLLMProvider,
        }
        # Store provider to implementation mappings
        self._provider_impl: dict[str, str] = {}

    @property
    def _error_class(self) -> type[exceptions.LLMError]:
        return exceptions.LLMError

    def _validate_item(self, item: Any) -> str:
        """Validate and possibly transform item before registration."""
        match item:
            case str() if item in self._implementations:
                return item
            case type() if issubclass(item, LLMProvider):
                return item.__name__
            case str():  # Accept any string since we're mapping providers
                return "litellm"  # Default to litellm implementation
            case _:
                msg = f"Invalid provider type: {type(item)}"
                raise exceptions.LLMError(msg)

    def __setitem__(self, key: str, value: str | type[LLMProvider]) -> None:
        """Map provider to implementation."""
        impl_name = self._validate_item(value)
        self._provider_impl[key] = impl_name
        super().__setitem__(key, impl_name)

    def create_provider(self, name: str, config: LLMConfig) -> LLMProvider:
        """Create a provider instance."""
        try:
            impl_name = self._provider_impl[name]
            impl_class = self._implementations[impl_name]
            return impl_class(config)
        except KeyError as exc:
            msg = f"Provider not found: {name}"
            raise exceptions.LLMError(msg) from exc


# Global registry instance
default_registry = ProviderRegistry()
