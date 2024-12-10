"""Callable execution utilities."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, TypeGuard

from llmling.core.log import get_logger
from llmling.utils.importing import import_callable


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


logger = get_logger(__name__)


def is_async_callable(obj: Any) -> TypeGuard[Callable[..., Awaitable[Any]]]:
    """Check if an object is an async callable."""
    return asyncio.iscoroutinefunction(obj)


async def execute_callable(import_path: str, **kwargs: Any) -> Any:
    """Execute a callable and return its result.

    Args:
        import_path: Dot-separated path to callable
        **kwargs: Arguments to pass to the callable

    Returns:
        Result of the callable execution

    Raises:
        ValueError: If import or execution fails
    """
    try:
        callable_obj = import_callable(import_path)
        logger.debug("Executing %r: kwargs=%s", callable_obj, kwargs)
        # Execute the callable
        if is_async_callable(callable_obj):
            result = await callable_obj(**kwargs)
        else:
            result = callable_obj(**kwargs)
    except Exception as exc:
        msg = f"Error executing callable {import_path}: {exc}"
        raise ValueError(msg) from exc
    else:
        return result


if __name__ == "__main__":
    import_callable("datetime.datetime.strftime")
