from __future__ import annotations

import pytest

from llmling.config.manager import ConfigManager
from llmling.processors.registry import ProcessorRegistry


@pytest.fixture
def config_manager(test_config):
    """Get config manager with test configuration."""
    return ConfigManager(test_config)


@pytest.fixture
def processor_registry():
    """Get clean processor registry."""
    return ProcessorRegistry()
