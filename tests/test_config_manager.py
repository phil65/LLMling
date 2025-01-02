"""Tests for configuration management."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from llmling.config.manager import ConfigManager
from llmling.config.models import Config, ProcessorConfig, TextResource, ToolConfig
from llmling.core import exceptions


if TYPE_CHECKING:
    from pathlib import Path


VERSION = "1.0"

VALID_CONFIG = """\
version: "1.0"

context_processors:
  test_processor:
    import_path: llmling.testing.processors.uppercase_text

resources:
  test-resource:
    type: text
    content: Test content
    description: Test resource

resource_groups:
  test-group:
    - test-resource

tools:
  test-tool:
    import_path: llmling.testing.tools.example_tool
    name: test
    description: Test tool
"""


@pytest.fixture
def valid_config() -> Config:
    """Create a valid configuration."""
    return Config.from_yaml(VALID_CONFIG)


@pytest.fixture
def config_file(tmp_path: Path, valid_config: Config) -> Path:
    """Create a test configuration file."""
    config_file = tmp_path / "config.yml"
    content = valid_config.model_dump(exclude_none=True)
    config_file.write_text(yaml.dump(content))
    return config_file


def test_load_config(config_file: Path) -> None:
    """Test loading configuration from file."""
    manager = ConfigManager.load(config_file)
    assert manager.config.version == VERSION
    assert "test-resource" in manager.config.resources
    assert "test_processor" in manager.config.context_processors


def test_validate_config(valid_config: Config) -> None:
    """Test configuration validation."""
    manager = ConfigManager(valid_config)
    # Should not raise
    manager.validate_or_raise()

    # Should have no warnings
    assert not manager.validate()


def test_validate_processor_config(valid_config: Config) -> None:
    """Test processor configuration validation."""
    manager = ConfigManager(valid_config)
    procs = manager.config.context_processors
    # Test missing import path
    # Now allowed by Pydantic, but caught by ConfigManager
    procs["invalid_import"] = ProcessorConfig(import_path="")

    # Test non-existent module
    procs["invalid_module"] = ProcessorConfig(import_path="non.existent")

    warnings = manager.validate()
    assert len(warnings) == 2  # noqa: PLR2004
    assert any("missing import_path" in w for w in warnings)
    assert any("Cannot import module" in w for w in warnings)


def test_validate_invalid_resource_group(valid_config: Config) -> None:
    """Test validation of invalid resource group references."""
    manager = ConfigManager(valid_config)

    # Add reference to non-existent resource
    manager.config.resource_groups["invalid"] = ["nonexistent"]

    warnings = manager.validate()
    assert warnings
    assert any("nonexistent" in w for w in warnings)


def test_validate_invalid_tool(valid_config: Config) -> None:
    """Test validation of invalid tool configuration."""
    manager = ConfigManager(valid_config)

    # Add invalid tool (missing import_path)
    cfg = ToolConfig(import_path="", name="invalid", description="Invalid tool")
    manager.config.tools["invalid"] = cfg

    warnings = manager.validate()
    assert warnings
    assert any("missing import_path" in w for w in warnings)


def test_validate_processor_references(valid_config: Config) -> None:
    """Test validation of processor references in resources."""
    manager = ConfigManager(valid_config)

    # Add resource with non-existent processor
    procs = [{"name": "nonexistent", "required": True}]
    manager.config.resources["invalid"] = TextResource(content="test", processors=procs)  # type: ignore

    warnings = manager.validate()
    assert warnings
    assert any("nonexistent" in w for w in warnings)


def test_load_invalid_config(tmp_path: Path) -> None:
    """Test loading invalid configuration."""
    invalid_file = tmp_path / "invalid.yml"
    invalid_file.write_text("invalid: : yaml: content")

    with pytest.raises(exceptions.ConfigError, match="Failed to load"):
        ConfigManager.load(invalid_file)


def test_save_config(tmp_path: Path, config_file: Path) -> None:
    """Test saving configuration."""
    manager = ConfigManager.load(config_file)

    save_path = tmp_path / "saved_config.yml"
    manager.save(save_path)

    # Load saved config and verify
    loaded = ConfigManager.load(save_path)
    assert loaded.config.model_dump() == manager.config.model_dump()


def test_save_invalid_config(tmp_path: Path, valid_config: Config) -> None:
    """Test saving invalid configuration."""
    manager = ConfigManager(valid_config)

    # Add reference to non-existent resource in group
    manager.config.resource_groups["invalid"] = ["nonexistent-resource"]

    save_path = tmp_path / "invalid_config.yml"

    # Should raise when trying to save invalid config
    with pytest.raises(exceptions.ConfigError):
        manager.save(save_path)

    # Should be able to save with validation disabled
    manager.save(save_path, validate=False)
    assert save_path.exists()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
