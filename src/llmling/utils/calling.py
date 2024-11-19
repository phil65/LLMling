"""Callable context loader."""

from __future__ import annotations

import asyncio
import importlib
from typing import TYPE_CHECKING, Any, TypeGuard


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


def is_async_callable(obj: Any) -> TypeGuard[Callable[..., Awaitable[Any]]]:
    """Check if an object is an async callable."""
    return asyncio.iscoroutinefunction(obj)


def import_callable(import_path: str) -> Callable[..., Any]:
    """Import a callable from an import path.

    Args:
        import_path: Dot-separated path to callable (e.g., "module.submodule.func")

    Returns:
        The imported callable

    Raises:
        ValueError: If import fails or object is not callable
    """
    try:
        module_path, callable_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        callable_obj = getattr(module, callable_name)

        if not callable(callable_obj):
            msg = f"Imported object {import_path} is not callable"
            raise ValueError(msg)  # noqa: TRY004
    except ImportError as exc:
        msg = f"Could not import callable: {import_path}"
        raise ValueError(msg) from exc
    except AttributeError as exc:
        msg = f"Could not find callable {import_path!r} "
        raise ValueError(msg) from exc
    else:
        return callable_obj


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
