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


def import_callable(path: str) -> Callable[..., Any]:
    """Import a callable from an import path.

    Args:
        path: Dot-separated path to callable (e.g., "module.submodule.func")

    Returns:
        The imported callable

    Raises:
        ValueError: If import fails or object is not callable
    """
    if not path:
        msg = "Import path cannot be empty"
        raise ValueError(msg)

    # Normalize path - replace colon with dot if present
    normalized_path = path.replace(":", ".")
    parts = normalized_path.split(".")

    # Try importing progressively smaller module paths
    for i in range(len(parts), 0, -1):
        try:
            # Try current module path
            module_path = ".".join(parts[:i])
            module = importlib.import_module(module_path)

            # Walk remaining parts as attributes
            obj = module
            for part in parts[i:]:
                obj = getattr(obj, part)

            # Check if we got a callable
            if callable(obj):
                return obj

            msg = f"Found object at {path} but it isn't callable"
            raise ValueError(msg)

        except ImportError:
            # Try next shorter path
            continue
        except AttributeError:
            # Attribute not found - try next shorter path
            continue

    # If we get here, no import combination worked
    msg = f"Could not import callable from path: {path}"
    raise ValueError(msg)


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


if __name__ == "__main__":
    import_callable("datetime.datetime.strftime")
