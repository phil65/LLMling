"""Utilities for working with prompt functions."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, get_type_hints

from docstring_parser import parse as parse_docstring

from llmling.core.log import get_logger
from llmling.prompts.models import ExtendedPromptArgument
from llmling.utils import importing


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from llmling.completions.types import CompletionFunction


logger = get_logger(__name__)


def extract_function_info(
    fn_or_path: str | Callable[..., Any],
    completions: Mapping[str, CompletionFunction | str] | None = None,
) -> tuple[list[ExtendedPromptArgument], str]:
    """Extract parameter info and description from a function.

    Example:
        >>> async def search_docs(query: str, max_results: int = 10) -> list[str]:
        ...     '''Search documentation.
        ...     Args:
        ...         query: Search string
        ...         max_results: Maximum number of results
        ...     '''
        ...
        >>> args, desc = extract_function_info(search_docs)
        >>> args
        [
            ExtendedPromptArgument(
                name="query",
                description="Search string",
                required=True,
                type_hint=str,
            ),
            ExtendedPromptArgument(
                name="max_results",
                description="Maximum number of results",
                required=False,
                type_hint=int,
                default=10,
            ),
        ]
        >>> desc
        'Search documentation.'

    Args:
        fn_or_path: Function or import path to analyze
        completions: Optional mapping of parameter names to completion functions
                    or their import paths

    Returns:
        Tuple of (list of argument definitions, function description)

    Raises:
        ValueError: If function cannot be imported or analyzed
    """
    try:
        # Import if needed
        fn = (
            importing.import_callable(fn_or_path)
            if isinstance(fn_or_path, str)
            else fn_or_path
        )

        # Get function metadata
        sig = inspect.signature(fn)
        hints = get_type_hints(fn, include_extras=True)

        # Parse docstring
        desc = f"Prompt from {fn.__name__}"
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
                ExtendedPromptArgument(
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
