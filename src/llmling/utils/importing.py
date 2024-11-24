"""Source code context loader."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import pkgutil
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from types import ModuleType


def get_module_source(
    import_path: str,
    recursive: bool = False,
    include_tests: bool = False,
) -> str:
    """Get source code from a module or package."""
    try:
        module = importlib.import_module(import_path)
        sources = list(
            _get_sources(
                module,
                recursive=recursive,
                include_tests=include_tests,
            )
        )
        return "\n\n# " + "-" * 40 + "\n\n".join(sources)

    except ImportError as exc:
        msg = f"Could not import module: {import_path}"
        raise ValueError(msg) from exc


def _get_sources(
    module: ModuleType,
    recursive: bool,
    include_tests: bool,
) -> Generator[str, None, None]:
    """Generate source code for a module and optionally its submodules."""
    # Get the module's source code
    if hasattr(module, "__file__") and module.__file__:
        path = Path(module.__file__)
        if _should_include_file(path, include_tests):
            yield f"# File: {path}\n{inspect.getsource(module)}"

    # If recursive and it's a package, get all submodules
    if recursive and hasattr(module, "__path__"):
        for _, name, _ in pkgutil.iter_modules(module.__path__):
            submodule_path = f"{module.__name__}.{name}"
            try:
                submodule = importlib.import_module(submodule_path)
                yield from _get_sources(submodule, recursive, include_tests)
            except ImportError:
                continue


def _should_include_file(path: Path, include_tests: bool) -> bool:
    """Check if a file should be included in the source."""
    if not include_tests:
        # Skip test files and directories
        parts = path.parts
        if any(p.startswith("test") for p in parts):
            return False
    return path.suffix == ".py"


def import_callable(path: str) -> Callable[..., Any]:
    """Import a callable from a dotted path.

    Supports both dot and colon notation:
    - Dot notation: module.submodule.Class.method
    - Colon notation: module.submodule:Class.method

    Examples:
        >>> import_callable("os.path.join")
        >>> import_callable("llmling.testing:processors.failing_processor")
        >>> import_callable("builtins.str.upper")
        >>> import_callable("sqlalchemy.orm:Session.query")

    Args:
        path: Import path using dots and/or colon

    Returns:
        Imported callable

    Raises:
        ValueError: If path cannot be imported or result isn't callable
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


if __name__ == "__main__":
    # Test both notations with various patterns
    test_cases = [
        # Dot notation
        "os.path.join",
        "builtins.str.upper",
        "llmling.testing.processors.failing_processor",
        "json.dumps",
        # Colon notation
        "llmling.testing:processors.failing_processor",
        "sqlalchemy.orm:Session.query",
        "django.db.models:Model.objects.filter",
        # Invalid cases to test error handling
        "nonexistent.module.function",
        "os.path.nonexistent",
        "",  # Empty path
    ]

    for test in test_cases:
        try:
            result = import_callable(test)
            print(f"✅ {test}: {result}")
        except ValueError as e:
            print(f"❌ {test}: {e}")
