from __future__ import annotations

import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Self

from upath import UPath

from llmling.core import exceptions
from llmling.core.log import get_logger


type Command = Sequence[str]

if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling.config.models import GlobalSettings


logger = get_logger(__name__)


class DependencyError(exceptions.ConfigError):
    """Error during dependency management."""


class DependencyManager:
    """Manages global tool dependencies."""

    def __init__(self, settings: GlobalSettings) -> None:
        self.settings = settings
        self._installed: set[str] = set()
        self._is_uv = self._detect_uv_environment()

    async def __aenter__(self) -> Self:
        """Set up dependencies on context entry."""
        await self.setup()
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Clean up on context exit."""
        # Currently no cleanup needed, but good for future

    def _detect_uv_environment(self) -> bool:
        """Detect if we're running in a uv environment."""
        try:
            # UV_VIRTUAL_ENV works on both platforms
            if "UV_VIRTUAL_ENV" in os.environ:
                return True

            # Check PATH for uv/uv.exe
            return bool(shutil.which("uv"))
        except Exception:  # noqa: BLE001
            return False

    def _get_pip_command(self) -> list[str]:
        """Get the appropriate pip command based on environment and settings."""
        if self.settings.prefer_uv or self._is_uv:
            # Check for uv in PATH - will find uv.exe on Windows
            if uv_path := shutil.which("uv"):
                return [str(uv_path), "pip"]
            if self.settings.prefer_uv:
                logger.warning("uv requested but not found, falling back to pip")

        # Use sys.executable for cross-platform compatibility
        # On Windows this will be 'python.exe'
        return [sys.executable, "-m", "pip"]

    def install_requirements(self) -> None:
        """Install missing requirements.

        Raises:
            DependencyError: If installation fails
        """
        if not self.settings.requirements:
            return

        missing = self.check_requirements()
        if not missing:
            return

        logger.info("Installing missing requirements: %s", missing)
        cmd = self._get_pip_command()
        cmd.append("install")

        if self.settings.pip_index_url:
            cmd.extend(["--index-url", self.settings.pip_index_url])

        cmd.extend(missing)

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            self._installed.update(missing)
            logger.debug("Package install output:\n%s", result.stdout)

        except subprocess.CalledProcessError as exc:
            msg = f"Failed to install requirements: {exc}\nOutput: {exc.stderr}"
            raise DependencyError(msg) from exc
        except Exception as exc:
            msg = f"Unexpected error installing requirements: {exc}"
            raise DependencyError(msg) from exc

    def ensure_importable(self, import_path: str) -> None:
        """Ensure a module can be imported."""
        try:
            module_name = import_path.split(".")[0]
            importlib.import_module(module_name)
        except ImportError as exc:
            installed = {dist.name for dist in importlib.metadata.distributions()}
            msg = (
                f"Module {module_name!r} not found. "
                f"Make sure it's included in global_settings.requirements "
                f"or the module path is in global_settings.extra_paths. "
                f"Currently installed packages: {', '.join(sorted(installed))}"
            )
            raise DependencyError(msg) from exc

    def check_requirements(self) -> list[str]:
        """Check which requirements need to be installed.

        Returns:
            List of requirements that are not yet installed
        """
        missing = []
        for req in self.settings.requirements:
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

    def update_python_path(self) -> None:
        """Add extra paths to Python path.

        Updates sys.path with extra paths from settings, ensuring they exist
        and are absolute paths.

        Logs warnings for invalid paths but doesn't raise exceptions.
        """
        if not self.settings.extra_paths:
            return

        for path in self.settings.extra_paths:
            try:
                # UPath handles path normalization cross-platform
                abs_path = UPath(path).resolve()
                if not abs_path.exists():
                    logger.warning("Path does not exist: %s", path)
                    continue

                # Convert to string in platform's format
                str_path = str(abs_path)
                if str_path not in sys.path:
                    sys.path.append(str_path)
                    logger.debug("Added %s to Python path", str_path)

            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to add path %s: %s", path, exc)

    def verify_paths(self, paths: Sequence[str | os.PathLike[str]]) -> None:
        """Verify that paths exist and are accessible.

        Args:
            paths: Sequence of paths to verify

        Raises:
            DependencyError: If any path is invalid or inaccessible
        """
        for path in paths:
            try:
                path_obj = UPath(path)
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

    def get_installed_requirements(self) -> list[str]:
        """Get list of requirements that were installed.

        Returns:
            List of installed requirement strings
        """
        return sorted(self._installed)

    def get_python_paths(self) -> list[str]:
        """Get current Python path entries.

        Returns:
            List of paths in sys.path
        """
        return sys.path.copy()

    async def setup(self) -> None:
        """Complete setup of dependencies.

        This is the main entry point that should be called to set up
        all dependencies. It:
        1. Installs missing requirements
        2. Updates Python path
        3. Verifies all paths exist

        Raises:
            DependencyError: If setup fails
        """
        try:
            self.install_requirements()
            self.update_python_path()
            if self.settings.extra_paths:
                self.verify_paths(self.settings.extra_paths)
        except Exception as exc:
            if isinstance(exc, DependencyError):
                raise
            msg = f"Dependency setup failed: {exc}"
            raise DependencyError(msg) from exc
