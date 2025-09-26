"""Utilities for working with prompt functions."""

from __future__ import annotations  # noqa: I001

from collections.abc import Iterator, Sequence
import inspect
import sys
from typing import TYPE_CHECKING, Any, get_type_hints, ForwardRef  # noqa: F401
from types import UnionType
from llmling.core import exceptions
from llmling.prompts.models import (
    BasePrompt,
    DynamicPrompt,
    FilePrompt,
    StaticPrompt,
)
from llmling.core.log import get_logger
from llmling.prompts.models import PromptParameter
from llmling.utils import importing
from collections.abc import Mapping


if TYPE_CHECKING:
    from collections.abc import Callable

    from llmling.completions.types import CompletionFunction


logger = get_logger(__name__)


def extract_function_info(
    fn_or_path: str | Callable[..., Any],
    completions: Mapping[str, CompletionFunction | str] | None = None,
) -> tuple[list[PromptParameter], str]:
    """Extract parameter info and description from a function.

    Args:
        fn_or_path: Function or import path to analyze
        completions: Optional mapping of parameter names to completion functions
                    or their import paths

    Returns:
        Tuple of (list of argument definitions, function description)

    Raises:
        ValueError: If function cannot be imported or analyzed
    """
    from docstring_parser import parse as parse_docstring

    try:
        # Import if needed
        fn = (
            importing.import_callable(fn_or_path)
            if isinstance(fn_or_path, str)
            else fn_or_path
        )

        # Get function metadata
        sig = inspect.signature(fn)
        module = sys.modules[fn.__module__]

        # Create a globals dict with common types
        globalns = {
            **module.__dict__,
            "Sequence": Sequence,
            "Iterator": Iterator,
            "Mapping": Mapping,
            "List": list,
            "Dict": dict,
            "Set": set,
            "Tuple": tuple,
            "Any": Any,
        }

        hints = get_type_hints(fn, include_extras=True, globalns=globalns)
        # Parse docstring
        desc = f"Prompt from {getattr(fn, '__name__', 'unknown')}"
        arg_docs = {}
        if docstring := inspect.getdoc(fn):
            parsed = parse_docstring(docstring)
            if parsed.short_description:
                desc = parsed.short_description
            arg_docs = {
                param.arg_name: param.description
                for param in parsed.params
                if param.arg_name and param.description
            }

        # Extract arguments
        completions = completions or {}
        args = []
        for name, param in sig.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            # Convert completion function to import path if present
            completion_path = None
            if name in completions:
                comp = completions[name]
                if isinstance(comp, str):
                    completion_path = comp
                else:
                    completion_path = f"{comp.__module__}.{comp.__qualname__}"

            args.append(
                PromptParameter(
                    name=name,
                    description=arg_docs.get(name),
                    required=param.default == param.empty,
                    type_hint=hints.get(name, Any),
                    default=None if param.default is param.empty else param.default,
                    completion_function=completion_path,
                )
            )

    except Exception as exc:
        msg = f"Failed to analyze function: {exc}"
        raise ValueError(msg) from exc
    else:
        return args, desc


def get_type_completions(arg: PromptParameter, current_value: str) -> list[str]:
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
            # Process the non-None type directly
            return get_type_completions(
                PromptParameter(
                    name=arg.name, type_hint=other_type, description=arg.description
                ),
                current_value,
            )

    # Handle bool
    if type_hint is bool:
        return ["true", "false"]

    return []


def get_description_completions(
    arg: PromptParameter,
    current_value: str,
) -> list[str]:
    """Get completions from argument description."""
    if not arg.description or "one of:" not in arg.description:
        return []

    try:
        options_part = arg.description.split("one of:", 1)[1]
        # Clean up options properly
        options = [opt.strip().rstrip(")") for opt in options_part.split(",")]
        return [opt for opt in options if opt]  # Remove empty strings
    except IndexError:
        return []


def to_prompt(item: Any) -> BasePrompt:
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


def get_completion_for_arg(arg: PromptParameter, current_value: str) -> list[str]:
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
        completions = [c for c in completions if str(c).lower().startswith(current_lower)]

    # Deduplicate while preserving order
    seen = set()
    return [x for x in completions if not (x in seen or seen.add(x))]  # type: ignore
