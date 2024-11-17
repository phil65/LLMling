"""Source code context loader."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import pkgutil
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Generator
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
