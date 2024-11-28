"""Package resources for LLMling configuration."""

from __future__ import annotations

import importlib.resources
from typing import Final

TEST_CONFIG: Final[str] = str(
    importlib.resources.files("llmling.config_resources") / "test.yml"
)
WEB_RESEARCH_CONFIG: Final[str] = str(
    importlib.resources.files("llmling.config_resources") / "web_research.yml"
)
WATCH_EXAMPLES_CONFIG: Final[str] = str(
    importlib.resources.files("llmling.config_resources") / "watch_examples.yml"
)

__all__ = [
    "TEST_CONFIG",
    "WATCH_EXAMPLES_CONFIG",
    "WEB_RESEARCH_CONFIG",
]
