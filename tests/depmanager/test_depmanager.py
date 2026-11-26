import importlib.metadata
import logging
import os
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest

from llmling.depmanager.exceptions import DependencyError
from llmling.depmanager.manager import DependencyManager


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path
    from unittest import mock


# Test data
SCRIPT_SINGLE_DEP = """\
# /// script
# dependencies = ["requests>=2.31.0"]
# ///
"""

SCRIPT_ANOTHER_DEP = """\
# /// script
# dependencies = ["pandas>=2.0.0"]
# ///
"""

SCRIPT_INVALID_PYTHON = """\
# /// script
# requires-python = ">=9999.0"  # Future Python!
# dependencies = ["requests"]
# ///
"""

SCRIPT_NO_DEPS = "print('test')"


@pytest.fixture
def temp_venv(tmp_path: Path) -> Generator[Path]:
    """Create a temporary virtual environment structure."""
    venv_path = tmp_path / "venv"
    venv_path.mkdir()
    (venv_path / "bin").mkdir()
    (venv_path / "lib").mkdir()
    os.environ["VIRTUAL_ENV"] = str(venv_path)
    try:
        yield venv_path
    finally:
        os.environ.pop("VIRTUAL_ENV", None)


class TestDependencyManager:
    def test_init(self, settings: dict[str, Any]) -> None:
        """Test initialization."""
        manager = DependencyManager(**settings)
        assert manager.extra_paths == ["."]
        assert not manager._installed

    def test_update_python_path(self, tmp_path: Path) -> None:
        """Test Python path updating."""
        path = str(tmp_path)
        original_path = sys.path.copy()

        try:
            manager = DependencyManager(extra_paths=[path])
            manager.update_python_path()

            assert path in sys.path
        finally:
            # Cleanup
            sys.path[:] = original_path

    async def test_setup(
        self,
        settings: dict[str, Any],
        mock_subprocess: mock.MagicMock,
        mock_importlib: mock.MagicMock,
    ) -> None:
        """Test complete setup."""
        mock_importlib.side_effect = importlib.metadata.PackageNotFoundError()

        async with DependencyManager(**settings) as manager:
            assert manager._installed
            mock_subprocess.assert_called_once()

    async def test_setup_with_nonexistent_script(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test setup with nonexistent script."""
        caplog.set_level(logging.WARNING)
        # Scripts that don't exist should be skipped with a warning
        async with DependencyManager(scripts=["/nonexistent/script.py"]) as manager:
            assert not manager.requirements

        # Check that a warning was logged
        assert "Script not found: /nonexistent/script.py" in caplog.text

    async def test_setup_with_invalid_script(self, tmp_path: Path) -> None:
        """Test setup with invalid Python script."""
        script = tmp_path / "invalid.py"
        script.write_text("This is not valid Python!", encoding="utf-8")

        with pytest.raises(DependencyError, match=r"Invalid Python syntax in script.*"):
            async with DependencyManager(scripts=[str(script)]):
                pass  # Should not reach this point

    async def test_cleanup(self, tmp_path: Path) -> None:
        """Test cleanup of temporary files."""
        script = tmp_path / "test.py"
        script.write_text(SCRIPT_NO_DEPS)

        manager = DependencyManager(scripts=[str(script)])
        scripts_dir = manager._scripts_dir

        assert scripts_dir.exists()
        manager.cleanup()
        assert not scripts_dir.exists()

    async def test_setup_with_scripts(self, tmp_path: Path) -> None:
        """Test setup with script loading."""
        script = tmp_path / "test.py"
        script.write_text(SCRIPT_SINGLE_DEP)

        async with DependencyManager(
            scripts=[str(script)],
            requirements=["pandas"],
        ) as manager:
            assert "requests>=2.31.0" in manager.requirements
            assert "pandas" in manager.requirements

    def test_integration(self, tmp_path: Path) -> None:
        """Integration test with real filesystem."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        manager = DependencyManager(
            requirements=["pip"],  # pip should always be available
            extra_paths=[str(test_dir)],
        )
        manager.install()  # Should not raise
        manager.update_python_path()
        assert str(test_dir) in sys.path

    def test_sync_context_manager(self, tmp_path: Path) -> None:
        """Test synchronous context manager."""
        script = tmp_path / "test.py"
        script.write_text(SCRIPT_SINGLE_DEP)  # Create test file first

        with DependencyManager(
            scripts=[str(script)],
            requirements=["pandas"],
        ) as manager:
            assert "requests>=2.31.0" in manager.requirements
            assert "pandas" in manager.requirements

    def test_sync_context_manager_error(
        self,
        mock_subprocess: mock.MagicMock,
        mock_importlib: mock.MagicMock,
    ) -> None:
        """Test synchronous context manager error handling."""
        mock_importlib.side_effect = importlib.metadata.PackageNotFoundError()
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, [], stderr="Error")

        with (
            pytest.raises(DependencyError, match="Failed to install requirements"),
            DependencyManager(requirements=["nonexistent-package"]),
        ):
            pass

    def test_direct_methods(self, tmp_path: Path) -> None:
        """Test direct install/uninstall methods."""
        script = tmp_path / "test.py"
        script.write_text(SCRIPT_SINGLE_DEP)

        manager = DependencyManager(
            scripts=[str(script)],
            requirements=["pandas"],
        )

        # Test installation
        manager.install()
        assert "requests>=2.31.0" in manager.requirements
        assert "pandas" in manager.requirements
        assert manager._scripts_dir.exists()

        # Test cleanup
        manager.uninstall()
        assert not manager._scripts_dir.exists()

    def test_direct_methods_error(
        self,
        mock_subprocess: mock.MagicMock,
        mock_importlib: mock.MagicMock,
    ) -> None:
        """Test error handling in direct methods."""
        mock_importlib.side_effect = importlib.metadata.PackageNotFoundError()
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, [], stderr="Error")

        manager = DependencyManager(requirements=["nonexistent-package"])
        with pytest.raises(DependencyError, match="Failed to install requirements"):
            manager.install()

    async def test_duplicate_module_names(self, tmp_path: Path) -> None:
        """Test handling of duplicate module names."""
        # Create two scripts with the same stem name in different directories
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        script1 = dir1 / "module.py"
        script2 = dir2 / "module.py"

        script1.write_text("print('script1')")
        script2.write_text("print('script2')")

        with pytest.raises(
            DependencyError,
            match=r"Duplicate module name 'module' from.*Already used by.*",
        ):
            async with DependencyManager(scripts=[str(script1), str(script2)]):
                pass

    def test_python_version_requirement_fail(self, tmp_path: Path) -> None:
        """Test Python version requirement validation."""
        script = tmp_path / "test.py"
        script.write_text(SCRIPT_INVALID_PYTHON)

        with (
            pytest.raises(
                DependencyError,
                match=r"requires Python >=9999.0, but current version is.*",
            ),
            DependencyManager(scripts=[str(script)]),
        ):
            pass

    def test_installed_requirements_tracking(
        self, tmp_path: Path, mock_importlib: mock.MagicMock
    ) -> None:
        """Test that all requirements are tracked, even if already installed."""
        # Create two test scripts
        script1 = tmp_path / "script1.py"
        script2 = tmp_path / "script2.py"
        script1.write_text(SCRIPT_SINGLE_DEP)  # has requests>=2.31.0
        script2.write_text(SCRIPT_ANOTHER_DEP)  # has pandas>=2.0.0

        # Mock that all packages are already installed
        mock_importlib.return_value = True

        man = DependencyManager(scripts=[str(script1), str(script2)])
        man.install()

        # Should track all requirements, regardless of whether they needed installation
        installed = man.get_installed_requirements()
        assert "requests>=2.31.0" in installed
        assert "pandas>=2.0.0" in installed
