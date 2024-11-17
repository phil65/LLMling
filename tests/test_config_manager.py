"""Tests for configuration management."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from llmling.config.manager import ConfigManager
from llmling.core import exceptions


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def config_content() -> str:
    """Create test configuration content."""
    return """
version: "1.0"
global_settings:
    timeout: 30
    max_retries: 3
    temperature: 0.7
context_processors: {}
llm_providers:
    test-provider:
        model: test/model
        name: Test
contexts:
    test-context:
        type: text
        content: "Test content"
        description: "Test context"
task_templates:
    test-template:
        provider: test-provider
        context: test-context
        settings:
            temperature: 0.8
provider_groups: {}
context_groups: {}
"""


@pytest.fixture
def config_file(tmp_path: Path, config_content: str) -> Path:
    """Create a test configuration file."""
    config_file = tmp_path / "config.yml"
    config_file.write_text(config_content)
    return config_file


def test_load_config(config_file: Path) -> None:
    """Test loading configuration from file."""
    manager = ConfigManager.load(config_file)
    assert manager.config.version == "1.0"
    assert manager.config.global_settings.timeout == 30
    assert "test-template" in manager.config.task_templates


def test_load_invalid_config(tmp_path: Path) -> None:
    """Test loading invalid configuration."""
    invalid_file = tmp_path / "invalid.yml"
    invalid_file.write_text("invalid: yaml: content")

    with pytest.raises(exceptions.ConfigError):
        ConfigManager.load(invalid_file)


def test_save_config(tmp_path: Path, config_file: Path) -> None:
    """Test saving configuration."""
    manager = ConfigManager.load(config_file)

    save_path = tmp_path / "saved_config.yml"
    manager.save(save_path)

    # Load saved config and verify
    loaded = ConfigManager.load(save_path)
    assert loaded.config.model_dump() == manager.config.model_dump()


def test_get_effective_settings(config_file: Path) -> None:
    """Test getting effective settings for a template."""
    manager = ConfigManager.load(config_file)
    settings = manager.get_effective_settings("test-template")

    assert settings["temperature"] == 0.8  # From template
    assert settings["timeout"] == 30  # From global
    assert settings["max_retries"] == 3  # From global


def test_validate_references(config_file: Path) -> None:
    """Test configuration reference validation."""
    from llmling.config.models import TaskTemplate

    manager = ConfigManager.load(config_file)
    warnings = manager.validate_references()
    assert not warnings  # Should be valid

    # Add invalid reference using proper TaskTemplate model
    manager.config.task_templates["invalid"] = TaskTemplate(
        provider="non-existent",
        context="non-existent",
    )

    warnings = manager.validate_references()
    assert len(warnings) == 2  # Should have provider and context warnings


if __name__ == "__main__":
    pytest.main(["-v", __file__])
