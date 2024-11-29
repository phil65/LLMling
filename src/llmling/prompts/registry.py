"""Registry for prompt templates."""

from __future__ import annotations

import logging
from types import UnionType
from typing import TYPE_CHECKING, Any

from llmling.core import exceptions
from llmling.core.baseregistry import BaseRegistry
from llmling.prompts.function import create_prompt_from_callable
from llmling.prompts.models import ExtendedPromptArgument, Prompt, PromptMessage


logger = logging.getLogger(__name__)


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

    async def get_completions(
        self,
        prompt_name: str,
        argument_name: str,
        current_value: str,
    ) -> list[str]:
        """Get completions for a prompt argument.

        This method tries different completion sources in order:
        1. Custom completion function (if provided)
        2. Type-based completions (Literal, bool, etc.)
        3. Description-based completions (from "one of:" syntax)
        4. Default value (if no current value)

        Args:
            prompt_name: Name of the prompt
            argument_name: Name of the argument
            current_value: Current input value

        Returns:
            List of possible completions

        Raises:
            LLMLingError: If prompt not found
        """
        # Handle unknown prompt
        try:
            prompt = self[prompt_name]
        except KeyError as exc:
            msg = f"Prompt not found: {prompt_name}"
            raise exceptions.LLMLingError(msg) from exc

        # Find matching argument (return empty list if not found)
        arg = next(
            (a for a in prompt.arguments if a.name == argument_name),
            None,
        )
        if not arg:
            msg = "Argument %s not found in prompt %s"
            logger.debug(msg, argument_name, prompt_name)
            return []

        completions: list[str] = []

        try:
            # 1. Try custom completion function
            if arg.completion_function:
                try:
                    if items := arg.completion_function(current_value):
                        completions.extend(str(item) for item in items)
                except Exception:
                    msg = "Custom completion failed for %s.%s"
                    logger.exception(msg, prompt_name, argument_name)

            # 2. Add type-based completions
            if type_completions := self._get_type_completions(arg, current_value):
                completions.extend(str(val) for val in type_completions)

            # 3. Add description-based suggestions
            if desc_completions := self._get_description_completions(arg, current_value):
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
            return [
                x
                for x in completions
                if not (x in seen or seen.add(x))  # type: ignore
            ]

        except Exception:
            logger.exception(
                "Completion failed for prompt=%s argument=%s",
                prompt_name,
                argument_name,
            )
            return []

    def _get_type_completions(
        self,
        arg: ExtendedPromptArgument,
        current_value: str,
    ) -> list[str]:
        """Get completions based on argument type."""
        from typing import Literal, Union, get_args, get_origin

        type_hint = arg.type_hint
        if not type_hint:
            return []

        # Handle Literal types directly
        if get_origin(type_hint) is Literal:
            return [str(val) for val in get_args(type_hint)]

        # Handle Union/Optional types
        if get_origin(type_hint) in (Union, UnionType):
            args = get_args(type_hint)
            # If one of the args is None, process the other type
            if len(args) == 2 and type(None) in args:  # noqa: PLR2004
                other_type = next(arg for arg in args if arg is not type(None))
                # Process the non-None type directly instead of using replace
                return self._get_type_completions(
                    ExtendedPromptArgument(
                        name=arg.name,
                        type_hint=other_type,
                        description=arg.description,
                    ),
                    current_value,
                )

        # Handle bool
        if type_hint is bool:
            return ["true", "false"]

        return []

    def _get_description_completions(
        self,
        arg: ExtendedPromptArgument,
        current_value: str,
    ) -> list[str]:
        """Get completions from argument description."""
        if not arg.description or "one of:" not in arg.description:
            return []

        try:
            options_part = arg.description.split("one of:", 1)[1]
            # Clean up options properly
            options = [
                opt.strip().rstrip(")")  # Remove trailing )
                for opt in options_part.split(",")
            ]
            return [opt for opt in options if opt]  # Remove empty strings
        except IndexError:
            return []
