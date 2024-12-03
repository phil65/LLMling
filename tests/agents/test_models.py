"""Tests for agent configuration models."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from llmling.agents.models import AgentDefinition, SystemPrompt


def test_valid_system_prompt():
    """Test valid system prompt configurations."""
    prompts = [
        SystemPrompt(type="text", value="You are a helpful agent"),
        SystemPrompt(type="function", value="mymodule.get_prompt"),
        SystemPrompt(type="template", value="Context: {data}"),
    ]
    for prompt in prompts:
        assert prompt.type in ("text", "function", "template")
        assert prompt.value


def test_invalid_system_prompt():
    """Test invalid system prompt configurations."""
    with pytest.raises(ValidationError):
        SystemPrompt(type="invalid", value="test")  # invalid not in Literal

    with pytest.raises(ValidationError):
        # missing required value
        SystemPrompt(type="text")  # type: ignore


def test_valid_agent_definition():
    """Test valid complete agent configuration."""
    config = {
        "responses": {
            "TestResponse": {
                "description": "Test response",
                "fields": {
                    "message": {
                        "type": "str",
                        "description": "A message",
                    },
                    "score": {
                        "type": "int",
                        "constraints": {"ge": 0, "le": 100},
                    },
                },
            },
        },
        "agents": {  # This should be a dict of agents
            "test_agent": {  # Key is the agent ID
                "name": "Test Agent",
                "description": "A test agent",
                "model": "openai:gpt-4",
                "model_settings": {},
                "result_model": "TestResponse",
                "system_prompts": [
                    {"type": "text", "value": "You are a test agent"},
                ],
            },
        },
    }
    agent_def = AgentDefinition.model_validate(config)
    assert agent_def.responses["TestResponse"].fields["score"].constraints == {
        "ge": 0,
        "le": 100,
    }


def test_missing_referenced_response():
    """Test referencing non-existent response model."""
    config = {
        "responses": {},
        "agent": {
            "name": "Test Agent",
            "model": "openai:gpt-4",
            "result_model": "NonExistentResponse",
            "system_prompts": [],
        },
    }
    with pytest.raises(ValidationError):
        AgentDefinition.model_validate(config)
