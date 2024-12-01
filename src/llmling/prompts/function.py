"""Convert Python functions to MCP prompts."""

from __future__ import annotations

import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    get_type_hints,
)

from llmling.core.log import get_logger
from llmling.core.typedefs import MessageContent
from llmling.prompts.models import ExtendedPromptArgument, Prompt, PromptMessage


if TYPE_CHECKING:
    from collections.abc import Callable

    from llmling.prompts.completion import CompletionFunction


logger = get_logger(__name__)


def create_prompt_from_callable(
    fn: Callable[..., Any] | str,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    template_override: str | None = None,
    completions: dict[str, CompletionFunction] | None = None,
) -> Prompt:
    """Create a prompt from a callable.

    Args:
        fn: Function or import path to create prompt from
        name_override: Optional override for prompt name
        description_override: Optional override for prompt description
        template_override: Optional override for message template
        completions: Optional dict mapping argument names to completion functions

    Returns:
        Prompt instance

    Raises:
        ValueError: If callable cannot be imported or is invalid
    """
    # Import if string path provided
    if isinstance(fn, str):
        from llmling.utils import importing

        fn = importing.import_callable(fn)

    # Get function metadata
    name = name_override or fn.__name__
    sig = inspect.signature(fn)
    hints = get_type_hints(fn, include_extras=True)

    # Get description from docstring
    doc = inspect.getdoc(fn)
    description = description_override or (
        doc.split("\n\n")[0] if doc else f"Prompt from {name}"
    )

    # Parse docstring for argument descriptions
    arg_docs = _parse_arg_docs(fn)
    completion_funcs = completions or {}

    # Create arguments
    arguments = []
    for param_name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        type_hint = hints.get(param_name, Any)
        required = param.default == param.empty
        arg_desc = arg_docs.get(param_name, "")

        arguments.append(
            ExtendedPromptArgument(
                name=param_name,
                description=arg_desc,
                required=required,
                type_hint=type_hint,  # Store original type hint
                default=None if param.default is param.empty else param.default,
                completion_function=completion_funcs.get(param_name),
            )
        )

    # Create message template. Will be formatted with function result
    template = template_override if template_override else "{result}"
    # Create prompt messages
    content = MessageContent(type="text", content=f"Content from {name}:\n")
    messages = [
        PromptMessage(role="system", content=content),
        PromptMessage(role="user", content=MessageContent(type="text", content=template)),
    ]
    path = f"{fn.__module__}.{fn.__qualname__}"
    return Prompt(
        name=name,
        description=description,
        arguments=arguments,
        messages=messages,
        metadata={"source": "function", "import_path": path},
    )


def _parse_arg_docs(fn: Callable[..., Any]) -> dict[str, str]:
    """Parse argument descriptions from docstring.

    Args:
        fn: Function to parse docstring from

    Returns:
        Dictionary mapping argument names to their descriptions
    """
    doc = inspect.getdoc(fn)
    if not doc:
        return {}

    arg_docs: dict[str, str] = {}
    lines = doc.split("\n")
    in_args = False
    current_arg = None

    for line in lines:
        line = line.strip()

        # Start of Args section
        if line == "Args:":
            in_args = True
            continue

        # End of Args section
        if in_args and (not line or line.startswith(("Returns:", "Raises:"))):
            break

        # Parse argument
        if in_args:
            if line and not line.startswith(" "):
                # New argument definition
                if ":" in line:
                    arg_name, desc = line.split(":", 1)
                    current_arg = arg_name.strip()
                    arg_docs[current_arg] = desc.strip()
            elif current_arg and line:
                # Continuation of previous argument description
                arg_docs[current_arg] += " " + line.strip()

    return arg_docs


if __name__ == "__main__":

    def example_function(text: str, mode: Literal["a", "b"] = "a") -> str:
        """Process text with given mode and tags.

        Args:
            text: Input text to process
            mode: Processing mode (one of: a, b)
        """
        return text

    prompt = create_prompt_from_callable(example_function)
    print(f"Created prompt: {prompt}")
    print(f"Arguments: {prompt.arguments}")
    print(f"Messages: {prompt.messages}")
