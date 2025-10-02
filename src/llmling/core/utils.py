"""Provides a decorator for exception handling with template support."""

from __future__ import annotations

import functools
import inspect
import string
import sys
from typing import TYPE_CHECKING

from llmling.core.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable


logger = get_logger(__name__)


def error_handler[R, **P](
    log_template: str,
    catch_exception: type[Exception],
    *,
    chain_with: type[Exception] | None = None,
    hide_internal_trace: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that logs function calls and handles exceptions.

    Args:
        log_template: Template string for logging (uses function parameters)
        catch_exception: Exception type to catch
        chain_with: Optional exception type to chain with the caught exception
        hide_internal_trace: Whether to hide decorator frames from traceback

    Returns:
        Decorated function that includes logging and exception handling
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Create a template with safe_substitute support
        template = string.Template(log_template)
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Combine args and kwargs using actual parameter names
            params = dict(zip(param_names, args))
            params.update(kwargs)

            try:
                # Use safe_substitute to handle missing parameters gracefully
                log_msg = template.safe_substitute(params)
                logger.info(log_msg)

                return func(*args, **kwargs)

            except catch_exception as exc:
                error_msg = f"Error in {func.__name__}: {exc}"

                if chain_with is not None:
                    new_exc = chain_with(error_msg)
                    if hide_internal_trace:
                        tb = sys.exc_info()[2]
                        if tb is not None and tb.tb_next is not None:
                            new_exc.__traceback__ = tb.tb_next
                    raise new_exc from exc
                raise

        return wrapper

    return decorator


if __name__ == "__main__":
    # Example usage
    import logging

    logging.basicConfig(level=logging.DEBUG)

    @error_handler(
        log_template="Processing item: ${item}",  # Note the $ prefix for template vars
        catch_exception=ValueError,
        chain_with=RuntimeError,
    )
    def process_item(item: str) -> str:
        if not item:
            msg = "Item cannot be empty"
            raise ValueError(msg)
        return item.upper()

    # Successful case
    result = process_item("test")
    print(f"Result: {result}")
    # Error case
    try:
        result = process_item("")
        print(f"Result: {result}")
    except RuntimeError as e:
        print(f"Caught error: {e}")
