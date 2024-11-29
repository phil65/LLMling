"""Registry for prompt templates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling.core import exceptions
from llmling.core.baseregistry import BaseRegistry
from llmling.prompts.function import create_prompt_from_callable
from llmling.prompts.models import Prompt, PromptMessage


if TYPE_CHECKING:
    from collections.abc import Callable


class PromptRegistry(BaseRegistry[str, Prompt]):
    """Registry for prompt templates."""

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

    def register_function(
        self,
        fn: Callable[..., Any] | str,
        name: str | None = None,
        *,
        replace: bool = False,
    ) -> None:
        """Register a function as a prompt."""
        prompt = create_prompt_from_callable(fn, name_override=name)
        self.register(prompt.name, prompt, replace=replace)

    async def get_messages(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> list[PromptMessage]:
        """Get formatted messages for a prompt."""
        prompt = self[name]
        return prompt.format(arguments or {})
