from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest import mock

import pytest


if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_subprocess() -> Generator[mock.MagicMock]:
    """Mock subprocess.run."""
    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Successfully installed pytest-7.0.0"
        yield mock_run


@pytest.fixture
def mock_importlib() -> Generator[mock.MagicMock]:
    """Mock importlib.metadata."""
    with mock.patch("importlib.metadata.distribution") as mock_dist:
        yield mock_dist


@pytest.fixture
def settings() -> dict[str, Any]:
    """Create test settings."""
    return dict(
        requirements=["pytest>=7.0.0", "mock>=5.0.0"],
        extra_paths=["."],
        prefer_uv=False,
    )
