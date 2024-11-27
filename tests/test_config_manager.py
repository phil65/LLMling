"""Tests for configuration management."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from llmling.config.manager import ConfigManager
from llmling.core import exceptions


if TYPE_CHECKING:
    from pathlib import Path


VERSION = "1.0"
EXPECTED_WARNING_COUNT = 2

CONFIG_YAML = f"""
version: "{VERSION}"
context_processors: {{}}
contexts:
    test-context:
        type: text
        content: "Test content"
        description: "Test context"

resource_groups: {{}}
"""


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Create a test configuration file."""
    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(CONFIG_YAML)
    return config_file


def test_load_config(config_file: Path):
    """Test loading configuration from file."""
    manager = ConfigManager.load(config_file)
    assert manager.config.version == VERSION


def test_load_invalid_config(tmp_path: Path):
    """Test loading invalid configuration."""
    invalid_file = tmp_path / "invalid.yml"
    _ = invalid_file.write_text("invalid: yaml: content")

    with pytest.raises(exceptions.ConfigError):
        _ = ConfigManager.load(invalid_file)


def test_save_config(tmp_path: Path, config_file: Path):
    """Test saving configuration."""
    manager = ConfigManager.load(config_file)

    save_path = tmp_path / "saved_config.yml"
    manager.save(save_path)

    # Load saved config and verify
    loaded = ConfigManager.load(save_path)
    assert loaded.config.model_dump() == manager.config.model_dump()


if __name__ == "__main__":
    _ = pytest.main(["-v", __file__])
