"""Tests for task management."""

from __future__ import annotations

from unittest import mock

import pytest

from llmling.config import Config, LLMProviderConfig, TaskTemplate
from llmling.core import exceptions
from llmling.task.executor import TaskExecutor
from llmling.task.manager import TaskManager


@pytest.fixture
def mock_config() -> Config:
    """Create a mock configuration."""
    return mock.MagicMock(spec=Config)


@pytest.fixture
def mock_executor() -> TaskExecutor:
    """Create a mock task executor."""
    return mock.MagicMock(spec=TaskExecutor)


def test_resolve_provider_direct(mock_config: mock.MagicMock) -> None:
    """Test direct provider resolution."""
    # Setup
    provider_config = LLMProviderConfig(
        name="Test Provider",
        model="test/model",
    )
    mock_config.llm_providers = {"test-provider": provider_config}
    mock_config.provider_groups = {}

    template = TaskTemplate(
        provider="test-provider",
        context="test-context",
    )

    manager = TaskManager(mock_config, mock.MagicMock())
    provider_name, config = manager._resolve_provider(template)

    assert provider_name == "test-provider"
    assert config == provider_config


def test_resolve_provider_group(mock_config: mock.MagicMock) -> None:
    """Test provider group resolution."""
    # Setup
    provider_config = LLMProviderConfig(
        name="Test Provider",
        model="test/model",
    )
    mock_config.llm_providers = {"test-provider": provider_config}
    mock_config.provider_groups = {"group1": ["test-provider"]}

    template = TaskTemplate(
        provider="group1",
        context="test-context",
    )

    manager = TaskManager(mock_config, mock.MagicMock())
    provider_name, config = manager._resolve_provider(template)

    assert provider_name == "test-provider"
    assert config == provider_config


def test_resolve_provider_not_found(mock_config: mock.MagicMock) -> None:
    """Test provider resolution failure."""
    mock_config.llm_providers = {}
    mock_config.provider_groups = {}

    template = TaskTemplate(
        provider="non-existent",
        context="test-context",
    )

    manager = TaskManager(mock_config, mock.MagicMock())
    with pytest.raises(exceptions.TaskError):
        manager._resolve_provider(template)
