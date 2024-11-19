"""Tests for task management."""

from __future__ import annotations

from unittest import mock

import pytest

from llmling.config import Config, LLMProviderConfig, TaskTemplate
from llmling.core import exceptions
from llmling.task.executor import TaskExecutor
from llmling.task.manager import TaskManager


@pytest.fixture
def mock_config() -> mock.MagicMock:
    """Create a mock configuration."""
    config = mock.MagicMock(spec=Config)
    # Set required attributes
    config.tools = {}  # Empty dict for no tools
    config.llm_providers = {}
    config.provider_groups = {}
    config.task_templates = {}
    config.contexts = {}
    config.context_groups = {}
    return config


@pytest.fixture
def mock_executor() -> TaskExecutor:
    """Create a mock task executor."""
    return mock.MagicMock(spec=TaskExecutor)


def test_resolve_provider_direct(mock_config: mock.MagicMock) -> None:
    """Test direct provider resolution."""
    # Setup
    provider_config = LLMProviderConfig(name="Test Provider", model="test/model")
    mock_config.llm_providers = {"test-provider": provider_config}
    template = TaskTemplate(provider="test-provider", context="test-context")
    manager = TaskManager(mock_config, mock.MagicMock())
    provider_name, config = manager._resolve_provider(template)

    assert provider_name == "test-provider"
    assert config == provider_config


def test_resolve_provider_group(mock_config: mock.MagicMock) -> None:
    """Test provider group resolution."""
    # Setup
    provider_config = LLMProviderConfig(name="Test Provider", model="test/model")
    mock_config.llm_providers = {"test-provider": provider_config}
    mock_config.provider_groups = {"group1": ["test-provider"]}

    template = TaskTemplate(provider="group1", context="test-context")

    manager = TaskManager(mock_config, mock.MagicMock())
    provider_name, config = manager._resolve_provider(template)

    assert provider_name == "test-provider"
    assert config == provider_config


def test_resolve_provider_not_found(mock_config: mock.MagicMock) -> None:
    """Test provider resolution failure."""
    # Config already has empty dicts from fixture

    template = TaskTemplate(provider="non-existent", context="test-context")

    manager = TaskManager(mock_config, mock.MagicMock())
    with pytest.raises(exceptions.TaskError):
        manager._resolve_provider(template)


# Additional tests for context resolution
def test_resolve_context_direct(mock_config: mock.MagicMock) -> None:
    """Test direct context resolution."""
    context = mock.MagicMock()
    mock_config.contexts = {"test-context": context}

    template = TaskTemplate(provider="test-provider", context="test-context")

    manager = TaskManager(mock_config, mock.MagicMock())
    result = manager._resolve_context(template)

    assert result == context


def test_resolve_context_group(mock_config: mock.MagicMock) -> None:
    """Test context group resolution."""
    context = mock.MagicMock()
    mock_config.contexts = {"test-context": context}
    mock_config.context_groups = {"group1": ["test-context"]}

    template = TaskTemplate(provider="test-provider", context="group1")

    manager = TaskManager(mock_config, mock.MagicMock())
    result = manager._resolve_context(template)

    assert result == context


def test_resolve_context_not_found(mock_config: mock.MagicMock) -> None:
    """Test context resolution failure."""
    template = TaskTemplate(provider="test-provider", context="non-existent")

    manager = TaskManager(mock_config, mock.MagicMock())
    with pytest.raises(exceptions.TaskError):
        manager._resolve_context(template)
