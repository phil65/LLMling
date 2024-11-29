"""Registry for prompt templates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling.core import exceptions
from llmling.core.baseregistry import BaseRegistry
from llmling.prompts.models import Prompt, PromptResult
from llmling.prompts.rendering import render_prompt


if TYPE_CHECKING:
    from llmling.processors.registry import ProcessorRegistry
    from llmling.resources.loaders.registry import ResourceLoaderRegistry


class PromptRegistry(BaseRegistry[str, Prompt]):
    """Registry for prompt templates."""

    def __init__(
        self,
        loader_registry: ResourceLoaderRegistry | None = None,
        processor_registry: ProcessorRegistry | None = None,
    ) -> None:
        """Initialize registry.

        Args:
            loader_registry: Optional registry for resolving resources
            processor_registry: Optional registry for content processors
        """
        super().__init__()
        self.loader_registry = loader_registry
        self.processor_registry = processor_registry

    @property
    def _error_class(self) -> type[exceptions.LLMLingError]:
        return exceptions.LLMLingError

    def _validate_item(self, item: Any) -> Prompt:
        """Validate and convert items to Prompt instances."""
        match item:
            case Prompt():
                return item
            case dict():
                return Prompt.model_validate(item)
            case _:
                msg = f"Invalid prompt type: {type(item)}"
                raise exceptions.LLMLingError(msg)

    async def render(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> PromptResult:
        """Render a prompt template with arguments."""
        prompt = self[name]
        return await render_prompt(
            prompt,
            arguments or {},
            loader_registry=self.loader_registry,
            processor_registry=self.processor_registry,
        )
