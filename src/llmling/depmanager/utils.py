"""Utility functions for dependency management."""

from __future__ import annotations

import ast
import importlib.metadata
import logging
import os
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

from upath import UPath as Path

from llmling.depmanager.exceptions import DependencyError
from llmling.depmanager.parser import parse_pep723_deps


if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike

    import upath

    type Command = list[str]

logger = logging.getLogger(__name__)


def verify_paths(paths: Sequence[str | PathLike[str]]) -> None:
    """Verify that paths exist and are accessible."""
    for path in paths:
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                msg = f"Path does not exist: {path}"
                raise DependencyError(msg)  # noqa: TRY301
            if not path_obj.is_dir():
                msg = f"Path is not a directory: {path}"
                raise DependencyError(msg)  # noqa: TRY301
        except Exception as exc:
            if isinstance(exc, DependencyError):
                raise
            msg = f"Invalid path {path}: {exc}"
            raise DependencyError(msg) from exc


def validate_script(content: str, script_path: str) -> None:
    """Verify script content is valid Python code.

    Args:
        content: The script content
        script_path: Path to script (for error messages)

    Raises:
        DependencyError: If script content has invalid syntax
    """
    try:
        ast.parse(content)
    except SyntaxError as exc:
        msg = f"Invalid Python syntax in script {script_path}: {exc}"
        raise DependencyError(msg) from exc


def detect_uv() -> bool:
    """Detect if we're running in a uv environment."""
    try:
        return "UV_VIRTUAL_ENV" in os.environ or bool(shutil.which("uv"))
    except Exception:  # noqa: BLE001
        return False


def get_pip_command(*, prefer_uv: bool = False, is_uv: bool = False) -> Command:
    """Get the appropriate pip command based on environment and settings."""
    if prefer_uv or is_uv:
        # Check for uv in PATH - will find uv.exe on Windows
        if uv_path := shutil.which("uv"):
            return [str(uv_path), "pip"]
        if prefer_uv:
            logger.warning("uv requested but not found, falling back to pip")

    # Use sys.executable for cross-platform compatibility
    return [sys.executable, "-m", "pip"]


async def collect_file_dependencies_async(path: str | PathLike[str]) -> set[str]:
    """Collect dependencies from a Python file asynchronously."""
    try:
        from upathtools import read_path

        content = await read_path(path, encoding="utf-8")
        return set(parse_pep723_deps(content))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to parse dependencies from %s: %s", path, exc)
        return set()


async def scan_directory_deps_async(directory: str | PathLike[str]) -> set[str]:
    """Recursively scan directory for PEP 723 dependencies asynchronously."""
    all_deps: set[str] = set()
    dir_path = Path(directory)

    # Don't scan site-packages or other system directories
    if "site-packages" in str(dir_path):
        return all_deps

    try:
        from upathtools import read_folder

        files = await read_folder(
            directory,
            pattern="*.py",
            recursive=True,
            exclude=["__pycache__/*", "*.pyc"],
        )

        for content in files.values():
            all_deps.update(parse_pep723_deps(content))

    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to scan %s for dependencies: %s", directory, exc)

    return all_deps


def collect_file_dependencies(path: str | PathLike[str] | upath.UPath) -> set[str]:
    """Collect dependencies from a Python file."""
    from upathtools import to_upath

    try:
        content = to_upath(path).read_text(encoding="utf-8", errors="ignore")
        return set(parse_pep723_deps(content))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to parse dependencies from %s: %s", path, exc)
        return set()


def scan_directory_deps(directory: str | PathLike[str]) -> set[str]:
    """Recursively scan directory for PEP 723 dependencies."""
    all_deps: set[str] = set()
    dir_path = Path(directory)

    # Don't scan site-packages or other system directories
    if "site-packages" in str(dir_path):
        return all_deps

    try:
        for path in dir_path.rglob("*.py"):
            all_deps.update(collect_file_dependencies(path))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to scan %s for dependencies: %s", directory, exc)
    return all_deps


def check_requirements(requirements: list[str]) -> list[str]:
    """Check which requirements need to be installed.

    Args:
        requirements: List of requirements to check

    Returns:
        List of requirements that are not yet installed
    """
    missing = []
    for req in requirements:
        try:
            # Split requirement into name and version specifier
            name = req.split(">=")[0].split("==")[0].split("<")[0].strip()
            importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(req)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error checking requirement %s: %s", req, exc)
            missing.append(req)
    return missing


def install_requirements(
    requirements: list[str],
    *,
    pip_command: Command | None = None,
    pip_index_url: str | None = None,
) -> set[str]:
    """Install missing requirements.

    Args:
        requirements: Requirements to install
        pip_command: Base pip command to use (default: pip)
        pip_index_url: Optional custom PyPI index URL

    Returns:
        Set of installed requirements

    Raises:
        DependencyError: If installation fails
    """
    pip_command = pip_command or ["pip"]
    if not requirements:
        return set()

    cmd = [*pip_command, "install"]

    if pip_index_url:
        cmd.extend(["--index-url", pip_index_url])

    cmd.extend(requirements)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug("Package install output:\n%s", result.stdout)
        return set(requirements)

    except subprocess.CalledProcessError as exc:
        msg = f"Failed to install requirements: {exc}\nOutput: {exc.stderr}"
        raise DependencyError(msg) from exc
    except Exception as exc:
        msg = f"Unexpected error installing requirements: {exc}"
        raise DependencyError(msg) from exc


def ensure_importable(import_path: str) -> None:
    """Ensure a module can be imported."""
    try:
        module_name = import_path.split(".", maxsplit=1)[0]
        importlib.import_module(module_name)
    except ImportError as exc:
        installed = {dist.name for dist in importlib.metadata.distributions()}
        msg = (
            f"Module {module_name!r} not found. "
            f"Make sure it's included in requirements "
            f"or the module path is in extra_paths. "
            f"Currently installed packages: {', '.join(sorted(installed))}"
        )
        raise DependencyError(msg) from exc


def in_virtualenv() -> bool:
    """Check if we're running inside a virtual environment or safe Jupyter kernel.

    Returns:
        True if running in a venv, conda env, or Jupyter kernel within a venv
    """
    # Basic virtualenv checks
    is_venv = (
        # Standard venv/virtualenv
        hasattr(sys, "real_prefix")
        # Python 3's venv
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        # Conda
        or bool(os.environ.get("CONDA_PREFIX"))
        # UV
        or bool(os.environ.get("UV_VIRTUAL_ENV"))
    )

    # If we're already in a venv, no need to check Jupyter
    if is_venv:
        return True

    # Only allow Jupyter if it's running in a virtual environment
    parts = Path(sys.argv[0]).parts
    in_jupyter = (
        hasattr(sys, "ps1")
        or "IPython" in sys.modules
        or os.environ.get("JUPYTER_RUNTIME_DIR") is not None
        or any(name.startswith(("jupyter-", "ipython")) for name in parts)
    )

    if in_jupyter:
        # Check if Jupyter is running in a venv
        jupyter_python = sys.executable
        jupyter_prefix = Path(jupyter_python).parent.parent
        system_prefix = Path(getattr(sys, "base_prefix", sys.prefix))
        return jupyter_prefix != system_prefix

    return False


def get_venv_info() -> dict[str, str | None]:
    """Get information about the current virtual environment."""
    return {
        "is_venv": str(in_virtualenv()),
        "venv_path": sys.prefix,
        "base_path": getattr(sys, "base_prefix", sys.prefix),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "uv_venv": os.environ.get("UV_VIRTUAL_ENV"),
    }
