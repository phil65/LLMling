"""Registry for prompt templates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling.completions.protocols import CompletionProvider
from llmling.core import exceptions
from llmling.core.baseregistry import BaseRegistry
from llmling.core.log import get_logger
from llmling.prompts.models import BasePrompt, DynamicPrompt
from llmling.prompts.utils import get_completion_for_arg, to_prompt


logger = get_logger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable


class PromptRegistry(BaseRegistry[str, BasePrompt], CompletionProvider):
    """Registry for prompt templates."""

    @property
    def _error_class(self) -> type[exceptions.LLMLingError]:
        return exceptions.LLMLingError

    def register(self, key: str, item: BasePrompt, replace: bool = False) -> None:
        """Register prompt with its key as name."""
        # Create copy with name set
        item = item.model_copy(update={"name": key})
        super().register(key, item, replace)

    def _validate_item(self, item: Any) -> BasePrompt:
        """Validate and convert items to BasePrompt instances."""
        return to_prompt(item)

    def register_function(
        self,
        fn: Callable[..., Any] | str,
        name: str | None = None,
        *,
        replace: bool = False,
    ) -> None:
        """Register a function as a prompt."""
        prompt = DynamicPrompt.from_callable(fn, name_override=name)
        assert prompt.name
        self.register(prompt.name, prompt, replace=replace)

    async def get_completions(
        self,
        current_value: str,
        argument_name: str | None = None,
        **options: Any,
    ) -> list[str]:
        """Get completions for a prompt argument."""
        try:
            prompt_name = options.get("prompt_name")
            if not prompt_name or not argument_name:
                return []

            prompt = self[prompt_name]
            arg = next((a for a in prompt.arguments if a.name == argument_name), None)
            if not arg:
                return []
            return get_completion_for_arg(arg, current_value)

        except Exception:
            logger.exception("Completion failed")
            return []
