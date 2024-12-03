"""Tests for agent configuration loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from llmling.agents.loader import load_agent_config_file
from llmling.agents.models import AgentDefinition


if TYPE_CHECKING:
    from pathlib import Path


def test_load_valid_config(valid_config: str):
    """Test loading valid configuration."""
    config = AgentDefinition.model_validate(valid_config)
    assert isinstance(config, AgentDefinition)
    assert config.agents["support"].name == "Support Agent"
    assert "SupportResult" in config.responses


def test_load_invalid_file():
    """Test loading non-existent file."""
    with pytest.raises(ValueError):  # noqa: PT011
        load_agent_config_file("nonexistent.yml")


def test_load_invalid_yaml(tmp_path: Path):
    """Test loading invalid YAML content."""
    invalid_file = tmp_path / "invalid.yml"
    invalid_file.write_text("invalid: yaml: content:")

    with pytest.raises(ValueError):  # noqa: PT011
        load_agent_config_file(invalid_file)
