"""Registry for prompt templates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling.completions.protocols import CompletionProvider
from llmling.core import exceptions
from llmling.core.baseregistry import BaseRegistry
from llmling.core.log import get_logger
from llmling.prompts.models import (
    BasePrompt,
    DynamicPrompt,
    FilePrompt,
    StaticPrompt,
)
from llmling.prompts.utils import get_description_completions, get_type_completions


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
        match item:
            case BasePrompt():
                return item
            case dict():
                if "type" not in item:
                    msg = "Missing prompt type in configuration"
                    raise ValueError(msg)
                match item["type"]:
                    case "text":
                        return StaticPrompt.model_validate(item)
                    case "function":
                        return DynamicPrompt.model_validate(item)
                    case "file":
                        return FilePrompt.model_validate(item)
                msg = f"Unknown prompt type: {item['type']}"
                raise ValueError(msg)
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

            completions: list[str] = []

            # 1. Try custom completion function
            if arg.completion_function:
                try:
                    if items := arg.completion_function(current_value):
                        completions.extend(str(item) for item in items)
                except Exception:
                    logger.exception("Custom completion failed")

            # 2. Add type-based completions
            if type_completions := get_type_completions(arg, current_value):
                completions.extend(str(val) for val in type_completions)

            # 3. Add description-based suggestions
            if desc_completions := get_description_completions(arg, current_value):
                completions.extend(str(val) for val in desc_completions)

            # 4. Add default if no current value
            if not current_value and arg.default is not None:
                completions.append(str(arg.default))

            # Filter by current value if provided
            if current_value:
                current_lower = current_value.lower()
                completions = [
                    c for c in completions if str(c).lower().startswith(current_lower)
                ]

            # Deduplicate while preserving order
            seen = set()
            return [x for x in completions if not (x in seen or seen.add(x))]  # type: ignore

        except Exception:
            logger.exception("Completion failed")
            return []
