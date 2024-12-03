from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def valid_config() -> dict[str, Any]:
    """Fixture providing valid agent configuration."""
    return {
        "responses": {
            "SupportResult": {
                "description": "Support agent response",
                "fields": {
                    "advice": {
                        "type": "str",
                        "description": "Support advice",
                    },
                    "risk": {
                        "type": "int",
                        "constraints": {"ge": 0, "le": 100},
                    },
                },
            },
            "ResearchResult": {
                "description": "Research agent response",
                "fields": {
                    "findings": {
                        "type": "str",
                        "description": "Research findings",
                    },
                },
            },
        },
        "agents": {
            "support": {
                "name": "Support Agent",
                "model": "openai:gpt-4",
                "model_settings": {
                    "retries": 3,
                    "result_retries": 2,
                },
                "result_model": "SupportResult",
                "system_prompts": [
                    {"type": "text", "value": "You are a support agent"},
                    {"type": "template", "value": "Context: {data}"},
                ],
            },
            "researcher": {
                "name": "Research Agent",
                "model": "openai:gpt-4",
                "result_model": "ResearchResult",
                "system_prompts": [
                    {"type": "text", "value": "You are a researcher"},
                ],
            },
        },
    }
