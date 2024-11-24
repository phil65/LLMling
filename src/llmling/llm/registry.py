"""LLM provider registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from epregistry import EntryPointRegistry

from llmling.core import exceptions
from llmling.core.baseregistry import BaseRegistry
from llmling.core.log import get_logger
from llmling.llm.base import LLMProvider
from llmling.llm.providers.dummy import DummyProvider
from llmling.llm.providers.litellmprovider import LiteLLMProvider


if TYPE_CHECKING:
    from llmling.llm.base import LLMConfig

logger = get_logger(__name__)


class ProviderFactory:
    """Factory to create configured provider instances."""

    def __init__(self, provider_class: type[LLMProvider]) -> None:
        """Initialize with provider class."""
        self.provider_class = provider_class

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.provider_class.__name__})"

    def create(self, config: LLMConfig) -> LLMProvider:
        """Create new provider instance with config."""
        return self.provider_class(config)


class ProviderRegistry(BaseRegistry[str, ProviderFactory]):
    """Registry for LLM provider factories."""

    BUILTIN_PROVIDERS: ClassVar[dict[str, type[LLMProvider]]] = {
        "litellm": LiteLLMProvider,
        "dummy": DummyProvider,
    }

    def __init__(self) -> None:
        """Initialize registry."""
        super().__init__()
        # Register default implementation
        self.register("litellm", ProviderFactory(LiteLLMProvider))
        self._ep_registry = EntryPointRegistry[type[LLMProvider]]("llmling.providers")

    def get_implementation(self, provider_type: str) -> type[LLMProvider]:
        """Get provider implementation class."""
        if provider_type in self.BUILTIN_PROVIDERS:
            return self.BUILTIN_PROVIDERS[provider_type]
        if provider_type in self._items:
            return self._items[provider_type].provider_class
        msg = f"Unknown provider type: {provider_type}"
        raise KeyError(msg)

    @property
    def _error_class(self) -> type[exceptions.LLMError]:
        return exceptions.LLMError

    def _validate_item(self, item: Any) -> ProviderFactory:
        """Validate and create factory from input."""
        try:
            match item:
                case type() if issubclass(item, LLMProvider):
                    return ProviderFactory(item)
                case ProviderFactory():
                    return item
                case str() if item in self.BUILTIN_PROVIDERS:
                    return ProviderFactory(self.BUILTIN_PROVIDERS[item])
                case _:
                    msg = f"Invalid provider type: {type(item)}"
                    raise exceptions.LLMError(msg)
        except TypeError as exc:
            msg = f"Invalid provider: {exc}"
            raise exceptions.LLMError(msg) from exc

    def load_providers(self) -> None:
        """Load all registered provider entry points."""
        for name, provider_class in self._ep_registry.load_all().items():
            try:
                self.register(name, ProviderFactory(provider_class))
                logger.debug("Loaded provider from entry point: %s", name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load provider %s: %s", name, exc)

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
default_registry.load_providers()
