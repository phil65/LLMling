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
from llmling.prompts.models import (
    ArgumentType,
    ExtendedPromptArgument,
    Prompt,
    PromptMessage,
)


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
    """Create a prompt from a callable or import path."""
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

    # Parse docstring for arg descriptions
    arg_docs = _parse_arg_docs(fn)

    # Create arguments
    arguments = []
    for param_name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        arg_type = hints.get(param_name, Any)
        required = param.default == param.empty
        arg_desc = arg_docs.get(param_name, "")

        arg_type, enum_values = _get_argument_type(arg_type)

        arguments.append(
            ExtendedPromptArgument(
                name=param_name,
                description=arg_desc,
                required=required,
                type=arg_type,
                enum_values=enum_values,
                default=None if param.default is param.empty else param.default,
            )
        )
    # Create message template
    if template_override:
        template = template_override
    else:
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
        PromptMessage(role="user", content=MessageContent(type="text", content=template)),
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
    """Parse argument descriptions from docstring."""
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


def _get_argument_type(type_hint: Any) -> tuple[ArgumentType, list[str] | None]:
    """Convert Python type hint to prompt argument type and possible values."""
    # Check for Literal types
    if get_origin(type_hint) is Literal:
        return ArgumentType.ENUM, [str(arg) for arg in get_args(type_hint)]

    # Check for bool
    if type_hint is bool:
        return ArgumentType.ENUM, ["true", "false"]

    # Handle Union/Optional types
    if get_origin(type_hint) in (Union, UnionType):
        args = get_args(type_hint)
        # If one of the args is None, process the other type
        if len(args) == 2 and type(None) in args:  # noqa: PLR2004
            other_type = next(arg for arg in args if arg is not type(None))
            return _get_argument_type(other_type)

    # Handle list, set, tuple
    if get_origin(type_hint) in (list, set, tuple):
        # We could potentially make this ENUM if we know the possible values
        return ArgumentType.TEXT, None

    # Default to TEXT for everything else
    return ArgumentType.TEXT, None


class _FunctionPrompt:
    """Converts Python functions to MCP prompts."""

    @classmethod
    def from_callable(
        cls,
        fn: Callable[..., Any],
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        template_override: str | None = None,
    ) -> Prompt:
        """Create a prompt from a callable."""
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

        # Create arguments
        arguments = []
        for param_name, param in sig.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            arg_type = hints.get(param_name, Any)
            required = param.default == param.empty
            arg_desc = arg_docs.get(param_name, "")

            # Use standalone function instead of static method
            arg_type, enum_values = _get_argument_type(arg_type)

            arguments.append(
                ExtendedPromptArgument(
                    name=param_name,
                    description=arg_desc,
                    required=required,
                    type=arg_type,
                    enum_values=enum_values,
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
                role="user", content=MessageContent(type="text", content=template)
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

    @staticmethod
    def _get_param_doc(fn: Callable[..., Any], param_name: str) -> str:
        """Extract parameter description from docstring."""
        if not fn.__doc__:
            return ""

        # Try to find parameter in docstring (Args: section)
        doc_lines = (inspect.getdoc(fn) or "").split("\n")
        in_args = False
        for line in doc_lines:
            if line.strip() == "Args:":
                in_args = True
                continue
            if in_args and line.strip().startswith(f"{param_name}:"):
                return line.split(":", 1)[1].strip()
        return ""


# Example usage
if __name__ == "__main__":

    def analyze_code(
        code: str, language: str = "python", focus: list[str] | None = None
    ) -> str:
        """Analyze code quality and structure.

        Args:
            code: Source code to analyze
            language: Programming language
            focus: Optional areas to focus on
        """
        return "abc"

    prompt = _FunctionPrompt.from_callable(analyze_code)
    print(f"Created prompt: {prompt}")
    print(f"Arguments: {prompt.arguments}")
    print(f"Messages: {prompt.messages}")
