"""Tests for configuration handling in LLMling."""

from __future__ import annotations

import pytest

from llmling import config
from llmling.config.runtime import RuntimeConfig


MINIMAL_CONFIG = """
# yaml-language-server: $schema=https://raw.githubusercontent.com/phil65/LLMling/refs/heads/main/schema/config-schema.json
toolsets:
  bird:
    type: openapi
    spec: "https://gist.githubusercontent.com/phil65/f6f3d4d459e206cc80ca0e2fb9e5c0d8/raw/739795ded6567740a6831c4b38e7bbef49544b82/bird_openapi.yml"
"""


def test_load_config_from_file() -> None:
    """Test loading configuration from a file."""
    cfg = config.Config.from_yaml(MINIMAL_CONFIG)
    with RuntimeConfig.from_config(cfg) as runtime:
        assert "findEntities" in runtime._tool_registry


if __name__ == "__main__":
    pytest.main([__file__])
