"""Convert Python functions to MCP prompts."""

from __future__ import annotations

import inspect
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from llmling.core.log import get_logger
from llmling.core.typedefs import MessageContent
from llmling.prompts.models import ExtendedPromptArgument, Prompt, PromptMessage
from llmling.utils import importing


if TYPE_CHECKING:
    from collections.abc import Callable


logger = get_logger(__name__)


def create_prompt_from_callable(
    fn: Callable[..., Any] | str,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    template_override: str | None = None,
) -> Prompt:
    """Create a prompt from a callable or import path.

    Args:
        fn: Function or import path to create prompt from
        name_override: Optional override for prompt name
        description_override: Optional override for tool description
        template_override: Optional override for message template

    Returns:
        Prompt instance

    Raises:
        ValueError: If callable cannot be imported or is invalid
    """
    # Import if string path provided
    if isinstance(fn, str):
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

    # Parse docstring for arg descriptions
    arg_docs = _parse_arg_docs(fn)

    # Create arguments with improved type handling
    arguments = []
    for param_name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        arg_type = hints.get(param_name, Any)
        required = param.default == param.empty
        arg_desc = arg_docs.get(param_name, "")

        # Get information about literal types if any
        literal_values = _get_literal_values(arg_type)
        if literal_values:
            arg_desc = f"{arg_desc} (one of: {', '.join(map(str, literal_values))})"

        arguments.append(
            ExtendedPromptArgument(
                name=param_name,
                description=arg_desc,
                required=required,
                type="text",  # Simplified to just use text type
                default=None if param.default is param.empty else param.default,
            )
        )

    # Create message template
    if template_override:
        template = template_override
    else:
        # Create default template from function signature
        arg_list = ", ".join(f"{arg.name}={{{arg.name}}}" for arg in arguments)
        template = f"Call {name}({arg_list})"

    # Create prompt messages
    messages = [
        PromptMessage(
            role="system",
            content=MessageContent(
                type="text",
                content=(
                    f"Function: {name}\n"
                    f"Description: {description}\n\n"
                    "Please provide the required arguments."
                ),
            ),
        ),
        PromptMessage(
            role="user",
            content=MessageContent(type="text", content=template),
        ),
    ]

    return Prompt(
        name=name,
        description=description,
        arguments=arguments,
        messages=messages,
        metadata={
            "source": "function",
            "import_path": f"{fn.__module__}.{fn.__qualname__}",
        },
    )


def _parse_arg_docs(fn: Callable[..., Any]) -> dict[str, str]:
    """Parse argument descriptions from docstring.

    Args:
        fn: Function to parse docstring from

    Returns:
        Dictionary mapping argument names to descriptions
    """
    doc = inspect.getdoc(fn)
    if not doc:
        return {}

    arg_docs: dict[str, str] = {}
    in_args = False
    current_arg = None

    for line in doc.split("\n"):
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


def _get_literal_values(typ: Any) -> list[Any] | None:
    """Get values from Literal type hint if present.

    Args:
        typ: Type hint to check

    Returns:
        List of literal values or None if not a Literal type
    """
    if get_origin(typ) is Literal:
        return list(get_args(typ))

    # Handle Union/Optional types
    if get_origin(typ) in (Union, UnionType):
        args = get_args(typ)
        # If one of the args is None, process the other type
        if len(args) == 2 and type(None) in args:  # noqa: PLR2004
            other_type = next(arg for arg in args if arg is not type(None))
            return _get_literal_values(other_type)

    return None


if __name__ == "__main__":
    # Example usage
    def analyze_code(
        code: str, language: str = "python", focus: list[str] | None = None
    ) -> str:
        """Analyze code quality and structure.

        Args:
            code: Source code to analyze
            language: Programming language
            focus: Optional areas to focus on
        """
        return "analysis result"

    prompt = create_prompt_from_callable(analyze_code)
    print(f"Created prompt: {prompt}")
    print(f"Arguments: {prompt.arguments}")
    messages = prompt.format({"code": "def example(): pass"})
    print(f"Formatted messages: {messages}")
