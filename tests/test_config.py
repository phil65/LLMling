"""Tests for configuration handling in LLMling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pydantic
import pytest
from upath import UPath
import yamling

from llmling import config
from llmling.processors.base import ProcessorConfig


if TYPE_CHECKING:
    import os


@pytest.fixture
def valid_config_dict() -> dict[str, Any]:
    """Create a valid configuration dictionary for testing."""
    test_yaml_path = (
        UPath(__file__).parent.parent / "src/llmling/config_resources/test.yml"
    )
    return yamling.load_yaml_file(test_yaml_path)


@pytest.fixture
def minimal_config_dict() -> dict[str, Any]:
    """Create a minimal valid configuration dictionary."""
    return {
        "version": "1.0",
        "global_settings": {
            "timeout": 30,
            "max_retries": 3,
            "temperature": 0.7,
        },
        "contexts": {
            "test-context": {
                "type": "text",
                "content": "test content",
                "description": "test description",
            }
        },
        "context_processors": {},
        "resource_groups": {},
        "task_templates": {
            "test-task": {
                "provider": "test-provider",
                "context": "test-context",
                "settings": {},
            }
        },
    }


def test_load_valid_config(valid_config_dict: dict[str, Any]) -> None:
    """Test loading a valid configuration."""
    cfg = config.Config.model_validate(valid_config_dict)
    assert cfg.version == "1.0"
    assert isinstance(cfg.global_settings, config.GlobalSettings)


def test_load_minimal_config(minimal_config_dict: dict[str, Any]) -> None:
    """Test loading a minimal valid configuration."""
    cfg = config.Config.model_validate(minimal_config_dict)
    assert cfg.version == "1.0"
    assert len(cfg.contexts) == 1


def test_validate_processor_config() -> None:
    """Test processor config validation."""
    # Test function processor
    with pytest.raises(pydantic.ValidationError):
        ProcessorConfig(type="function")  # Missing required import_path

    # Test template processor
    with pytest.raises(pydantic.ValidationError):
        ProcessorConfig(type="template")  # Missing required template

    # Test valid configs
    assert ProcessorConfig(type="function", import_path="test.func")
    assert ProcessorConfig(type="template", template="{{ content }}")


def test_validate_context_references(valid_config_dict: dict[str, Any]) -> None:
    """Test validation of context references in configuration."""
    # Modify config to include invalid context reference
    invalid_config = valid_config_dict.copy()
    invalid_config["resource_groups"] = {"invalid-group": ["non-existent-context"]}

    with pytest.raises(pydantic.ValidationError) as exc_info:
        config.Config.model_validate(invalid_config)
    assert "Resource non-existent-context" in str(exc_info.value)


def test_validate_source_context() -> None:
    """Test validation of source context configurations."""
    invalid = {"type": "source", "import_path": "invalid.1path", "description": "test"}
    with pytest.raises(pydantic.ValidationError) as exc_info:
        config.SourceResource.model_validate(invalid)
    assert "Invalid import path" in str(exc_info.value)

    valid = {"type": "source", "import_path": "valid.path", "description": "test"}
    ctx = config.SourceResource.model_validate(valid)
    assert ctx.import_path == "valid.path"


def test_validate_callable_context() -> None:
    """Test validation of callable context configurations."""
    invalid = {"type": "callable", "import_path": "invalid.1path", "description": "test"}
    with pytest.raises(pydantic.ValidationError) as exc_info:
        config.CallableResource.model_validate(invalid)
    assert "Invalid import path" in str(exc_info.value)

    valid = {"type": "callable", "import_path": "valid.path", "description": "test"}
    ctx = config.CallableResource.model_validate(valid)
    assert ctx.import_path == "valid.path"


def test_load_config_from_file(tmp_path: os.PathLike[str]) -> None:
    """Test loading configuration from a file."""
    config_path = UPath(tmp_path) / "test_config.yml"
    config_path.write_text(
        """
version: "1.0"
global_settings:
    timeout: 30
    max_retries: 3
    temperature: 0.7
context_processors: {}
contexts:
    test-context:
        type: text
        content: test content
        description: test description
resource_groups: {}
"""
    )

    cfg = config.load_config(config_path)
    assert isinstance(cfg, config.Config)
    assert cfg.version == "1.0"
    assert "test-context" in cfg.contexts


if __name__ == "__main__":
    pytest.main([__file__])
