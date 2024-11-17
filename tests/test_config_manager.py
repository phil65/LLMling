"""Tests for configuration management."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from llmling.config.manager import ConfigManager
from llmling.core import exceptions


if TYPE_CHECKING:
    from pathlib import Path


VERSION = "1.0"
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.7
TEMPLATE_TEMPERATURE = 0.8
EXPECTED_WARNING_COUNT = 2

CONFIG_YAML = f"""
version: "{VERSION}"
global_settings:
    timeout: {DEFAULT_TIMEOUT}
    max_retries: {DEFAULT_MAX_RETRIES}
    temperature: {DEFAULT_TEMPERATURE}
context_processors: {{}}
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
            temperature: {TEMPLATE_TEMPERATURE}
provider_groups: {{}}
context_groups: {{}}
"""


@pytest.fixture
def config_content() -> str:
    """Create test configuration content."""
    return CONFIG_YAML


@pytest.fixture
def config_file(tmp_path: Path, config_content: str) -> Path:
    """Create a test configuration file."""
    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(config_content)
    return config_file


def test_load_config(config_file: Path) -> None:
    """Test loading configuration from file."""
    manager = ConfigManager.load(config_file)
    assert manager.config.version == VERSION
    assert manager.config.global_settings.timeout == DEFAULT_TIMEOUT
    assert "test-template" in manager.config.task_templates


def test_load_invalid_config(tmp_path: Path) -> None:
    """Test loading invalid configuration."""
    invalid_file = tmp_path / "invalid.yml"
    _ = invalid_file.write_text("invalid: yaml: content")

    with pytest.raises(exceptions.ConfigError):
        _ = ConfigManager.load(invalid_file)


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

    assert settings["temperature"] == TEMPLATE_TEMPERATURE  # From template
    assert settings["timeout"] == DEFAULT_TIMEOUT  # From global
    assert settings["max_retries"] == DEFAULT_MAX_RETRIES  # From global


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
    assert (
        len(warnings) == EXPECTED_WARNING_COUNT
    )  # Should have provider and context warnings


if __name__ == "__main__":
    _ = pytest.main(["-v", __file__])
