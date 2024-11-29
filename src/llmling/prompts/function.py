from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, get_type_hints

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
    """Create a prompt from a callable or import path.

    Args:
        fn: Function or import path to create prompt from
        name_override: Optional override for prompt name
        description_override: Optional override for description
        template_override: Optional override for message template

    Returns:
        Prompt instance

    Raises:
        ValueError: If callable cannot be imported or is invalid
    """
    # Import if string path provided
    if isinstance(fn, str):
        from llmling.utils import importing

        fn = importing.import_callable(fn)

    return _FunctionPrompt.from_callable(
        fn,
        name_override=name_override,
        description_override=description_override,
        template_override=template_override,
    )


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
        """Create a prompt from a callable.

        Args:
            fn: Function to convert
            name_override: Optional override for prompt name
            description_override: Optional override for prompt description
            template_override: Optional override for message template

        Returns:
            MCP Prompt instance

        Example:
            >>> def analyze_code(
            ...     code: str,
            ...     language: str = "python",
            ...     focus: list[str] | None = None
            ... ) -> str:
            ...     '''Analyze code with given focus areas.'''
            ...     pass
            >>> prompt = FunctionPrompt.from_callable(analyze_code)
        """
        # Get function metadata
        name = name_override or fn.__name__
        description = description_override or inspect.getdoc(fn) or f"Prompt from {name}"
        sig = inspect.signature(fn)
        hints = get_type_hints(fn, include_extras=True)

        # Create arguments from parameters
        arguments = []
        for param_name, param in sig.parameters.items():
            # Skip *args and **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            arg_type = hints.get(param_name, Any)
            required = param.default == param.empty
            description = cls._get_param_doc(fn, param_name)

            arguments.append(
                ExtendedPromptArgument(
                    name=param_name,
                    description=description,
                    required=required,
                    type=cls._get_argument_type(arg_type),
                    default=None if param.default is param.empty else param.default,
                )
            )

        # Create message template
        if template_override:
            template = template_override
        else:
            # Create default template from function signature
            arg_list = ", ".join(f"{arg.name}={{{arg.name}}}" for arg in arguments)
            template = f"Execute {name}({arg_list})"

        # Create prompt messages
        messages = [
            PromptMessage(
                role="system",
                content=MessageContent.text(
                    f"Function: {name}\n"
                    f"Description: {description}\n\n"
                    "Please provide the required arguments."
                ),
            ),
            PromptMessage(role="user", content=MessageContent.text(template)),
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

    @staticmethod
    def _get_argument_type(type_hint: Any) -> ArgumentType:
        """Convert Python type hint to prompt argument type."""
        # Map Python types to ArgumentType
        if hasattr(type_hint, "__origin__"):  # Generic types
            origin = type_hint.__origin__
            if origin in (list, set, tuple):
                return ArgumentType.ENUM
        # Add more type mappings as needed
        return ArgumentType.TEXT


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
