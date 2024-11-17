"""Callable context loader."""

from __future__ import annotations

import asyncio
import importlib
from typing import TYPE_CHECKING, Any, TypeGuard


if TYPE_CHECKING:
    from collections.abc import Callable


def is_async_callable(obj: Any) -> TypeGuard[Callable[..., Any]]:
    """Check if an object is an async callable."""
    return asyncio.iscoroutinefunction(obj)


async def execute_callable(import_path: str, **kwargs: Any) -> str:
    """Execute a callable and return its result as a string."""
    try:
        # Import the callable
        module_path, callable_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        callable_obj = getattr(module, callable_name)

        # Execute the callable
        if is_async_callable(callable_obj):
            result = await callable_obj(**kwargs)
        else:
            result = callable_obj(**kwargs)

        # Convert result to string
        if isinstance(result, str):
            return result
        if isinstance(result, list | dict | set | tuple):
            import json

            return json.dumps(result, indent=2, default=str)
        return str(result)

    except ImportError as exc:
        msg = f"Could not import callable: {import_path}"
        raise ValueError(msg) from exc
    except (TypeError, ValueError) as exc:
        msg = f"Error executing callable {import_path}: {exc}"
        raise ValueError(msg) from exc
