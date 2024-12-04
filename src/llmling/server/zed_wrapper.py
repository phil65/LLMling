"""Function parameter encoding utils. Used for Zed to overcome one-argument limitation."""

from __future__ import annotations

from functools import wraps
import inspect
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable


P = ParamSpec("P")
R = TypeVar("R")


def with_encoded_params(
    func: Callable[P, R],
) -> Callable[P, R] | Callable[[str], R]:
    """Wrap a function to accept encoded parameters as a single string.

    Only wraps functions with more than one parameter. Functions with 0 or 1
    parameters are returned as-is.

    Format: "arg :: key1=value1 | key2=value2"
    The part before :: is the first argument, after are key-value pairs.

    Args:
        func: The function to wrap.

    Returns:
        Either the original function (if 0-1 params) or a wrapped version.
    """
    sig = inspect.signature(func)
    param_count = len(sig.parameters)

    # Don't wrap functions with 0 or 1 parameters
    if param_count <= 1:
        return func

    @wraps(func)
    def wrapper(param_string: str) -> R:
        # Split arg and kwargs
        parts = param_string.split(" :: ", 1)
        first_arg = parts[0]

        if len(parts) == 1:
            return func(first_arg)  # type: ignore

        # Parse kwargs
        kwargs: dict[str, Any] = {}
        for pair in parts[1].split(" | "):
            if not pair:
                continue
            key, value = pair.split("=", 1)
            # Convert value to appropriate type
            match value.lower():
                case "true":
                    kwargs[key] = True
                case "false":
                    kwargs[key] = False
                case "null":
                    kwargs[key] = None
                case _:
                    try:
                        if "." in value:
                            kwargs[key] = float(value)
                        else:
                            kwargs[key] = int(value)
                    except ValueError:
                        kwargs[key] = value

        return func(first_arg, **kwargs)  # type: ignore

    # Create new signature and docstring for the wrapper
    orig_sig = inspect.signature(func)
    orig_doc = func.__doc__ or ""

    # Create a new docstring that includes both wrapper and original info
    wrapper.__doc__ = f"""Wrapped version of {func.__name__}.

    Args:
        param_string: Encoded parameters in format: "arg :: key=value | key2=value2"
                     where 'arg' is passed as the first argument and the rest
                     are passed as keyword arguments.

    Original function:
        {orig_doc}

    Original signature:
        {func.__name__}{orig_sig}
    """

    # Update the wrapper's signature
    wrapper.__signature__ = inspect.Signature(  # type: ignore
        [
            inspect.Parameter(
                "param_string",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=str,
            )
        ]
    )

    return wrapper


if __name__ == "__main__":

    @with_encoded_params
    def process_item(
        item: str,
        color: str = "red",
        count: int = 1,
        active: bool = True,
    ) -> str:
        return f"Processing {count} {color} {item}(s), active: {active}"

    # Examples
    result = process_item("box :: color=blue | count=3")
    result = process_item("container :: active=false | count=5 | color=green")
    result = process_item("sphere")  # just arg, no kwargs
