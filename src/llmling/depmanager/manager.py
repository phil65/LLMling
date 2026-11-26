"""Core dependency management functionality."""

import logging
import shutil
import sys
import tempfile
from typing import TYPE_CHECKING, Self

import upath

from llmling.depmanager.exceptions import DependencyError, ScriptError
from llmling.depmanager.parser import check_python_version, parse_script_metadata
from llmling.depmanager.utils import (
    check_requirements,
    detect_uv,
    get_pip_command,
    get_venv_info,
    in_virtualenv,
    install_requirements,
    scan_directory_deps,
    scan_directory_deps_async,
    validate_script,
    verify_paths,
)


if TYPE_CHECKING:
    import os
    import types


from upath import UPath as Path


logger = logging.getLogger(__name__)


class DependencyManager:
    """Manages Python package dependencies."""

    def __init__(
        self,
        requirements: list[str] | None = None,
        *,
        prefer_uv: bool = False,
        extra_paths: list[str] | None = None,
        scripts: list[str] | None = None,
        pip_index_url: str | None = None,
        force_install: bool = False,
    ) -> None:
        """Initialize dependency manager.

        Args:
            requirements: List of package requirements
            prefer_uv: Whether to prefer uv over pip for package installation
            extra_paths: Additional Python paths to add
            scripts: List of script files to process for dependencies
            pip_index_url: Custom PyPI index URL
            force_install: Allow installing without being in a virtual environment

        Raises:
            DependencyError: If not in a virtual environment and force_install=False
        """
        self.prefer_uv = prefer_uv
        self.requirements = requirements or []
        self.extra_paths = extra_paths or []
        self.pip_index_url = pip_index_url
        self.force_install = force_install
        self._installed: set[str] = set()
        self._is_uv = detect_uv()
        self.scripts = scripts or []
        self._scripts_dir = Path(tempfile.mkdtemp(prefix="llmling_scripts_"))
        self._module_map: dict[str, str] = {}  # Maps module names to file paths

    def _check_venv(self) -> None:
        """Check if we're in a virtual environment.

        Raises:
            DependencyError: If not in a virtual environment and force_install=False
        """
        if not in_virtualenv() and not self.force_install:
            msg = (
                "Not running in a virtual environment. Installing packages globally "
                "is not recommended. Use force_install=True to override."
            )
            raise DependencyError(msg)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(prefer_uv={self.prefer_uv}, "
            f"requirements={self.requirements}, extra_paths={self.extra_paths}, "
            f"pip_index_url={self.pip_index_url})"
        )

    def __enter__(self) -> Self:
        """Set up dependencies on context entry."""
        self.setup()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Clean up on context exit."""
        self.cleanup()

    async def __aenter__(self) -> Self:
        """Set up dependencies on async context entry."""
        await self.async_setup()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Clean up on async context exit."""
        self.cleanup()

    def get_environment_info(self) -> dict[str, str | None]:
        """Get information about the current Python environment."""
        return {
            **get_venv_info(),
            "python_version": sys.version,
            "python_path": sys.executable,
            "is_uv": str(self._is_uv),
            "pip_index": self.pip_index_url,
        }

    def install(self) -> None:
        """Install dependencies and set up environment.

        A simpler alternative to the context manager. Does the same setup
        but requires manual cleanup via uninstall().

        Raises:
            DependencyError: If setup fails
        """
        self.setup()

    def uninstall(self) -> None:
        """Clean up installed dependencies and temporary files."""
        self.cleanup()

    def install_dependency(self, requirement: str) -> None:
        """Install a single package dependency.

        Args:
            requirement: Package requirement specification (e.g., "requests>=2.28.0")

        Raises:
            DependencyError: If installation fails or not in a virtual environment
        """
        self._check_venv()
        # Check if already installed
        if not check_requirements([requirement]):
            logger.debug("Requirement %r is already satisfied", requirement)
            self._installed.add(requirement)
            return

        logger.info("Installing requirement: %s", requirement)
        pip_cmd = get_pip_command(prefer_uv=self.prefer_uv, is_uv=self._is_uv)

        try:
            install_requirements(
                [requirement],
                pip_command=pip_cmd,
                pip_index_url=self.pip_index_url,
            )
            self._installed.add(requirement)
            logger.info("Successfully installed: %s", requirement)
        except Exception as exc:
            if isinstance(exc, DependencyError):
                raise
            msg = f"Failed to install {requirement}: {exc}"
            raise DependencyError(msg) from exc

    def install_script(self, script_path: str | os.PathLike[str]) -> None:
        """Install dependencies from a PEP 723 script.

        Args:
            script_path: Path to the Python script

        Raises:
            DependencyError: If script processing or installation fails
            ScriptError: If script metadata is invalid
        """
        path = Path(script_path)

        try:
            # Read and validate script
            content = path.read_text("utf-8", errors="ignore")
            validate_script(content, str(path))

            # Parse metadata
            metadata = parse_script_metadata(content)

            # Check Python version constraint if specified
            if metadata.python_version:
                check_python_version(metadata.python_version, str(path))

            # Install dependencies
            if not metadata.dependencies:
                logger.debug("No dependencies found in script: %s", path)
                return
            msg = "Installing %d dependencies from script %s"
            logger.info(msg, len(metadata.dependencies), path)

            for req in metadata.dependencies:
                self.install_dependency(req)

        except FileNotFoundError as exc:
            msg = f"Script not found: {path}"
            raise DependencyError(msg) from exc
        except Exception as exc:
            if isinstance(exc, DependencyError | ScriptError):
                raise
            msg = f"Failed to process script {path}: {exc}"
            raise DependencyError(msg) from exc

    def _setup_script_modules(self) -> None:
        """Set up importable modules from scripts."""
        for script_path in self.scripts:
            logger.debug("Processing script: %s", script_path)
            try:
                content = Path(script_path).read_text("utf-8", errors="ignore")
                metadata = parse_script_metadata(content)
                validate_script(content, script_path)
                # Check Python version first
                if metadata.python_version:
                    logger.debug("Found Python constraint: %s", metadata.python_version)
                    check_python_version(metadata.python_version, script_path)

                # Add dependencies
                if metadata.dependencies:
                    logger.debug("Found dependencies: %s", metadata.dependencies)
                    self.requirements.extend(metadata.dependencies)

                # Extract base module name from filename
                base_name = Path(script_path).stem

                # Check for name collision
                if base_name in self._module_map:
                    msg = (
                        f"Duplicate module name '{base_name}' from {script_path}. "
                        f"Already used by {self._module_map[base_name]}"
                    )
                    raise DependencyError(msg)  # noqa: TRY301

                # Save to temporary location
                module_file = self._scripts_dir / f"{base_name}.py"
                module_file.write_text(content)

                # Map module name to file
                self._module_map[base_name] = str(module_file)

            except FileNotFoundError:
                logger.warning("Script not found: %s", script_path)
            except Exception as exc:
                if isinstance(exc, DependencyError):
                    raise
                msg = f"Failed to process script {script_path}: {exc}"
                logger.warning(msg)

        # Add scripts directory to Python path
        if self._scripts_dir and self._module_map:  # Only if we have valid scripts
            sys.path.insert(0, str(self._scripts_dir))

    def verify_import_path(self, import_path: str) -> None:
        """Verify that an import path matches available modules."""
        module_name = import_path.split(".", maxsplit=1)[0]
        if module_name not in self._module_map:
            msg = (
                f"Import path {import_path!r} references unknown module. "
                f"Available modules: {', '.join(sorted(self._module_map))}"
            )
            raise DependencyError(msg)

    def update_python_path(self) -> None:
        """Add extra paths to Python path."""
        if not self.extra_paths:
            return

        for path in self.extra_paths:
            try:
                abs_path = Path(path).resolve()
                if not abs_path.exists():
                    logger.warning("Path does not exist: %s", path)
                    continue
                if (str_path := str(abs_path)) not in sys.path:
                    sys.path.append(str_path)
                    logger.debug("Added %s to Python path", str_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to add path %s: %s", path, exc)

    def get_installed_requirements(self) -> list[str]:
        """Get list of requirements that were installed."""
        return sorted(self._installed)

    def get_python_paths(self) -> list[str]:
        """Get current Python path entries."""
        return sys.path.copy()

    def _setup_requirements(self, requirements: set[str]) -> None:
        """Handle requirements installation and environment setup."""
        # Update requirements with all found dependencies
        self.requirements = sorted(requirements)

        # Find which requirements need installation
        missing = check_requirements(self.requirements)

        # Check venv status only if we have missing requirements to install
        if missing:
            self._check_venv()
            logger.info("Installing missing requirements: %s", missing)
            pip_cmd = get_pip_command(prefer_uv=self.prefer_uv, is_uv=self._is_uv)
            install_requirements(
                missing,
                pip_command=pip_cmd,
                pip_index_url=self.pip_index_url,
            )
            logger.info("Successfully installed: %s", missing)

        # Track all requirements that were handled
        self._installed.update(self.requirements)

        # Update Python path
        self.update_python_path()

        # Verify paths exist
        if self.extra_paths:
            logger.debug("Verifying paths: %s", self.extra_paths)
            verify_paths(self.extra_paths)

    def setup(self) -> None:
        """Complete setup of dependencies."""
        try:
            # First set up script modules to collect their dependencies
            self._setup_script_modules()

            # Collect all dependencies (explicit + PEP 723)
            requirements = set(self.requirements)

            # Add PEP 723 requirements from extra paths
            for path in self.extra_paths:
                if Path(path).is_dir() and (new_deps := scan_directory_deps(path)):
                    logger.debug("Found dependencies in %s: %s", path, new_deps)
                    requirements.update(new_deps)

            self._setup_requirements(requirements)

        except Exception as exc:
            self.cleanup()  # Ensure cleanup on error
            if isinstance(exc, DependencyError):
                raise
            msg = f"Dependency setup failed: {exc}"
            raise DependencyError(msg) from exc

    async def async_setup(self) -> None:
        """Complete async setup of dependencies."""
        try:
            # First set up script modules to collect their dependencies
            self._setup_script_modules()

            # Collect all dependencies (explicit + PEP 723)
            requirements = set(self.requirements)

            # Add PEP 723 requirements from extra paths
            for path in self.extra_paths:
                if Path(path).is_dir() and (new_deps := await scan_directory_deps_async(path)):
                    logger.debug("Found dependencies in %s: %s", path, new_deps)
                    requirements.update(new_deps)

            self._setup_requirements(requirements)

        except Exception as exc:
            self.cleanup()  # Ensure cleanup on error
            if isinstance(exc, DependencyError):
                raise
            msg = f"Dependency setup failed: {exc}"
            raise DependencyError(msg) from exc

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self._scripts_dir and self._scripts_dir.exists():
            if isinstance(self._scripts_dir, upath.UPath):
                self._scripts_dir.fs.rm(self._scripts_dir.path, recursive=True)
            else:
                shutil.rmtree(self._scripts_dir)
