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
    test_yaml_path = UPath(__file__).parent.parent / "src/llmling/resources/test.yml"
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
        "context_processors": {},
        "llm_providers": {
            "test-provider": {
                "model": "openai/test-model",
                "name": "Test provider",
            }
        },
        "provider_groups": {},
        "contexts": {
            "test-context": {
                "type": "text",
                "content": "test content",
                "description": "test description",
            }
        },
        "context_groups": {},
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
    assert len(cfg.llm_providers) > 0


def test_load_minimal_config(minimal_config_dict: dict[str, Any]) -> None:
    """Test loading a minimal valid configuration."""
    cfg = config.Config.model_validate(minimal_config_dict)
    assert cfg.version == "1.0"
    assert len(cfg.llm_providers) == 1
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


def test_validate_llm_provider() -> None:
    """Test validation of LLM provider configurations."""
    # Test invalid model format
    invalid = {"model": "invalid-model-format", "name": "test"}
    with pytest.raises(pydantic.ValidationError) as exc_info:
        config.LLMProviderConfig.model_validate(invalid)
    assert "must be in format 'provider/model'" in str(exc_info.value)

    # Test valid model format
    valid = {"model": "openai/gpt-4", "name": "test"}
    provider = config.LLMProviderConfig.model_validate(valid)
    assert provider.model == "openai/gpt-4"


def test_validate_context_references(valid_config_dict: dict[str, Any]) -> None:
    """Test validation of context references in configuration."""
    # Modify config to include invalid context reference
    invalid_config = valid_config_dict.copy()
    invalid_config["context_groups"] = {"invalid-group": ["non-existent-context"]}

    with pytest.raises(pydantic.ValidationError) as exc_info:
        config.Config.model_validate(invalid_config)
    assert "Context non-existent-context" in str(exc_info.value)


def test_validate_provider_references(valid_config_dict: dict[str, Any]) -> None:
    """Test validation of provider references in configuration."""
    # Modify config to include invalid provider reference
    invalid_config = valid_config_dict.copy()
    invalid_config["provider_groups"] = {"invalid-group": ["non-existent-provider"]}

    with pytest.raises(pydantic.ValidationError) as exc_info:
        config.Config.model_validate(invalid_config)
    assert "Provider non-existent-provider" in str(exc_info.value)


def test_validate_source_context() -> None:
    """Test validation of source context configurations."""
    invalid = {"type": "source", "import_path": "invalid.1path", "description": "test"}
    with pytest.raises(pydantic.ValidationError) as exc_info:
        config.SourceContext.model_validate(invalid)
    assert "Invalid import path" in str(exc_info.value)

    valid = {"type": "source", "import_path": "valid.path", "description": "test"}
    ctx = config.SourceContext.model_validate(valid)
    assert ctx.import_path == "valid.path"


def test_validate_callable_context() -> None:
    """Test validation of callable context configurations."""
    invalid = {"type": "callable", "import_path": "invalid.1path", "description": "test"}
    with pytest.raises(pydantic.ValidationError) as exc_info:
        config.CallableContext.model_validate(invalid)
    assert "Invalid import path" in str(exc_info.value)

    valid = {"type": "callable", "import_path": "valid.path", "description": "test"}
    ctx = config.CallableContext.model_validate(valid)
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
llm_providers:
    test-provider:
        name: Test provider
        model: openai/test-model
provider_groups: {}
contexts:
    test-context:
        type: text
        content: test content
        description: test description
context_groups: {}
task_templates:
    test-task:
        provider: test-provider
        context: test-context
        settings: {}
"""
    )

    cfg = config.load_config(config_path)
    assert isinstance(cfg, config.Config)
    assert cfg.version == "1.0"
    assert "test-provider" in cfg.llm_providers
    assert "test-context" in cfg.contexts


def test_task_template_validation(minimal_config_dict: dict[str, Any]) -> None:
    """Test validation of task template configurations."""
    # Test invalid provider reference
    invalid_config = minimal_config_dict.copy()
    # This exists in minimal_config
    task = {"provider": "non-existent-provider", "context": "test-context"}
    invalid_config["task_templates"]["invalid-task"] = task

    with pytest.raises(pydantic.ValidationError) as exc_info:
        config.Config.model_validate(invalid_config)
    assert "Provider non-existent-provider" in str(exc_info.value)

    # Test invalid context reference
    invalid_config = minimal_config_dict.copy()
    # First add a valid provider
    provider = {"model": "openai/test-model", "name": "test"}
    invalid_config["llm_providers"]["test-provider"] = provider
    task = {"provider": "test-provider", "context": "non-existent-context"}
    invalid_config["task_templates"]["invalid-task"] = task

    with pytest.raises(pydantic.ValidationError) as exc_info:
        config.Config.model_validate(invalid_config)
    assert "Context non-existent-context" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
