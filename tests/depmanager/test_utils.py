import importlib
import importlib.metadata
import os
import subprocess
import sys
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from llmling.depmanager.exceptions import DependencyError
from llmling.depmanager.utils import (
    check_requirements,
    collect_file_dependencies,
    detect_uv,
    ensure_importable,
    get_pip_command,
    get_venv_info,
    in_virtualenv,
    install_requirements,
    scan_directory_deps,
    verify_paths,
)


if TYPE_CHECKING:
    from pathlib import Path


# Test data
SCRIPT_WITH_DEPS = """\
# /// script
# dependencies = ["requests>=2.31.0", "pandas>=2.0.0"]
# requires-python = ">=3.12"
# ///
"""

SCRIPT_SINGLE_DEP = """\
# /// script
# dependencies = ["requests>=2.0.0"]
# ///
"""

SCRIPT_OTHER_DEP = """\
# /// script
# dependencies = ["pandas>=1.0.0"]
# ///
"""

SCRIPT_INVALID_TOML = """\
# /// script
# dependencies = ["incomplete
# ///
"""

SCRIPT_NO_DEPS = "print('hello')"


class TestUtils:
    def test_detect_uv(self) -> None:
        """Test UV detection."""
        with mock.patch.dict(os.environ, {"UV_VIRTUAL_ENV": "/path/to/venv"}):
            assert detect_uv()

        with mock.patch("shutil.which", return_value="/usr/local/bin/uv"):
            assert detect_uv()

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("shutil.which", return_value=None),
        ):
            assert not detect_uv()

    @pytest.mark.parametrize(
        ("platform", "uv_cmd"),
        [
            ("win32", "uv.exe"),
            ("linux", "uv"),
            ("darwin", "uv"),
        ],
    )
    def test_get_pip_command_uv(self, platform: str, uv_cmd: str) -> None:
        """Test pip command with UV."""
        with (
            mock.patch("sys.platform", platform),
            mock.patch("shutil.which", return_value=f"/usr/local/bin/{uv_cmd}"),
        ):
            cmd = get_pip_command(prefer_uv=True)
            assert cmd[0].endswith(uv_cmd)
            assert cmd[1] == "pip"

    @pytest.mark.parametrize(
        ("platform", "python_cmd"),
        [
            ("win32", "python.exe"),
            ("linux", "python3"),
            ("darwin", "python3"),
        ],
    )
    def test_get_pip_command_fallback(self, platform: str, python_cmd: str) -> None:
        """Test fallback pip command."""
        with (
            mock.patch("sys.platform", platform),
            mock.patch("shutil.which", return_value=None),
        ):
            cmd = get_pip_command(prefer_uv=False)
            assert cmd == [sys.executable, "-m", "pip"]

    def test_check_requirements(self, mock_importlib: mock.MagicMock) -> None:
        """Test requirement checking."""
        mock_importlib.side_effect = importlib.metadata.PackageNotFoundError()
        reqs = ["pytest>=7.0.0", "mock>=5.0.0"]

        missing = check_requirements(reqs)
        assert missing == reqs
        assert mock_importlib.call_count == len(reqs)

    def test_install_requirements(
        self, mock_subprocess: mock.MagicMock, mock_importlib: mock.MagicMock
    ) -> None:
        """Test requirement installation."""
        reqs = ["pytest>=7.0.0"]
        pip_cmd = [sys.executable, "-m", "pip"]

        installed = install_requirements(reqs, pip_command=pip_cmd)
        assert installed == set(reqs)

        cmd = mock_subprocess.call_args[0][0]
        assert "install" in cmd
        assert all(req in cmd for req in reqs)

    def test_install_requirements_failure(self, mock_subprocess: mock.MagicMock) -> None:
        """Test installation failure."""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, [], stderr="Error")

        with pytest.raises(DependencyError, match="Failed to install requirements"):
            install_requirements(
                ["nonexistent-package"],
                pip_command=[sys.executable, "-m", "pip"],
            )

    def test_verify_paths(self, tmp_path: Path) -> None:
        """Test path verification."""
        path = tmp_path / "test_dir"
        path.mkdir()

        # Valid path should not raise
        verify_paths([path])

        # Non-existent path should raise
        with pytest.raises(DependencyError, match="Path does not exist"):
            verify_paths([tmp_path / "nonexistent"])

        # File (not directory) should raise
        file_path = tmp_path / "test.txt"
        file_path.touch()
        with pytest.raises(DependencyError, match="Path is not a directory"):
            verify_paths([file_path])

    def test_collect_file_dependencies(self, tmp_path: Path) -> None:
        """Test dependency collection from file."""
        script = tmp_path / "test.py"
        script.write_text(SCRIPT_SINGLE_DEP)

        deps = collect_file_dependencies(script)
        assert deps == {"requests>=2.0.0"}

    def test_scan_directory_deps(self, tmp_path: Path) -> None:
        """Test directory scanning for dependencies."""
        # Create test files
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "mod1.py").write_text(SCRIPT_SINGLE_DEP)
        (tmp_path / "pkg" / "mod2.py").write_text(SCRIPT_OTHER_DEP)

        deps = scan_directory_deps(tmp_path)
        assert deps == {"requests>=2.0.0", "pandas>=1.0.0"}

    def test_collect_file_no_deps(self, tmp_path: Path) -> None:
        """Test dependency collection from file without deps."""
        script = tmp_path / "test.py"
        script.write_text(SCRIPT_NO_DEPS)

        deps = collect_file_dependencies(script)
        assert not deps

    def test_collect_file_invalid_toml(self, tmp_path: Path) -> None:
        """Test dependency collection with invalid TOML."""
        script = tmp_path / "test.py"
        script.write_text(SCRIPT_INVALID_TOML)

        # Should log error but not raise
        deps = collect_file_dependencies(script)
        assert not deps

    def test_ensure_importable(self) -> None:
        """Test import checking."""
        # Should work for stdlib modules
        ensure_importable("os.path")

        with pytest.raises(DependencyError, match="not found"):
            ensure_importable("nonexistent.module")

    def test_in_virtualenv(self) -> None:
        """Test virtual environment detection."""
        # Test when definitely not in a venv
        with (
            mock.patch("sys.prefix", "/usr"),
            mock.patch("sys.base_prefix", "/usr"),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            assert not in_virtualenv()

        # Test when in a venv (using UV as example)
        dct = {"UV_VIRTUAL_ENV": "/uv"}
        with mock.patch.dict(os.environ, dct, clear=True):
            assert in_virtualenv()

    def test_get_venv_info(self) -> None:
        """Test getting virtual environment information."""
        env_vars = {"VIRTUAL_ENV": "/venv", "UV_VIRTUAL_ENV": "/uv"}

        with mock.patch.dict(os.environ, env_vars, clear=True):
            info = get_venv_info()
            assert isinstance(info, dict)
            assert "is_venv" in info
            assert info["venv_path"] == sys.prefix

    def test_get_environment_info(self) -> None:
        """Test environment info includes all required fields."""
        from llmling.depmanager.manager import DependencyManager

        dm = DependencyManager(force_install=True)
        info = dm.get_environment_info()

        required_fields = {
            "is_venv",
            "venv_path",
            "base_path",
            "python_version",
            "python_path",
            "is_uv",
        }
        assert all(field in info for field in required_fields)
