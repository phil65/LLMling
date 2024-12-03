"""Tests for agent factory functionality."""

from __future__ import annotations

import typing
from typing import Any

from pydantic import BaseModel
from pydantic_ai import models
import pytest

from llmling import config_resources
from llmling.agents import LLMlingAgent
from llmling.agents.factory import (
    _create_response_model,
    _parse_type_annotation,
    create_agents_from_config,
)
from llmling.agents.models import AgentDefinition, ResponseDefinition, ResponseField
from llmling.config.runtime import RuntimeConfig
from llmling.core import exceptions


@pytest.fixture
def runtime() -> RuntimeConfig:
    """Fixture providing a RuntimeConfig."""
    from llmling.config.models import Config

    config = Config.from_file(config_resources.TEST_CONFIG)  # Use a test config
    return RuntimeConfig.from_config(config)


def test_parse_type_annotation():
    """Test string type parsing."""
    assert _parse_type_annotation("str") is str
    assert _parse_type_annotation("int") is int
    assert _parse_type_annotation("list[str]") == typing.List[str]  # noqa: UP006

    with pytest.raises(ValueError, match="invalid_type"):
        _parse_type_annotation("invalid_type")


def test_create_response_model():
    """Test dynamic model creation with proper field validation."""
    definition = ResponseDefinition(
        description="Test response",
        fields={
            "message": ResponseField(
                type="str",
                description="A test message",
            ),
            "count": ResponseField(
                type="int",
                constraints={"ge": 0, "le": 100},
                description="A count value",
            ),
            "items": ResponseField(
                type="list[str]",
                description="List of items",
            ),
        },
    )

    model = _create_response_model("TestResponse", definition)
    assert issubclass(model, BaseModel)
    assert model.__doc__ == "Test response"

    # Test field definitions
    message_field = model.model_fields["message"]
    assert message_field.annotation is str
    assert message_field.description == "A test message"
    assert not message_field.metadata  # No constraints

    # Test numeric constraints in metadata
    count_field = model.model_fields["count"]
    assert count_field.annotation is int
    assert count_field.description == "A count value"

    # Find constraint objects in metadata
    metadata_constraints = {type(m).__name__: m for m in count_field.metadata}
    assert "Ge" in metadata_constraints
    assert metadata_constraints["Ge"].ge == 0
    assert "Le" in metadata_constraints
    assert metadata_constraints["Le"].le == 100  # noqa: PLR2004

    # Test list field
    items_field = model.model_fields["items"]
    assert items_field.annotation == typing.List[str]  # noqa: UP006
    assert items_field.description == "List of items"

    # Test validation works
    instance = model(
        message="test",
        count=50,
        items=["a", "b"],
    )
    assert instance.message == "test"  # type: ignore
    assert instance.count == 50  # type: ignore  # noqa: PLR2004
    assert instance.items == ["a", "b"]  # type: ignore


def test_create_agents(valid_config: dict[str, Any], runtime: RuntimeConfig):
    """Test creation of multiple agents."""
    config = AgentDefinition.model_validate(valid_config)
    agents = create_agents_from_config(config, runtime)

    # Test we got both agents
    assert len(agents) == 2  # noqa: PLR2004
    assert "support" in agents
    assert "researcher" in agents

    # Test support agent configuration
    support = agents["support"]
    assert isinstance(support, LLMlingAgent)

    # Test core attributes
    assert support._name == "Support Agent"
    assert support.runtime == runtime

    # Test underlying PydanticAgent configuration
    pydantic_agent = support.pydantic_agent
    assert isinstance(pydantic_agent.model, models.Model)
    assert pydantic_agent.model.name() == "openai:gpt-4"
    assert not pydantic_agent._override_model
    assert not pydantic_agent._override_deps

    # Test model settings
    assert pydantic_agent._default_retries == 3  # noqa: PLR2004
    assert pydantic_agent._max_result_retries == 2  # noqa: PLR2004

    # Test system prompts
    assert len(pydantic_agent._system_prompts) == 2  # noqa: PLR2004
    assert pydantic_agent._system_prompts == (
        "You are a support agent",
        "Context: {data}",
    )
    assert not pydantic_agent._system_prompt_functions  # No dynamic prompts

    # Test result configuration
    result_schema = pydantic_agent._result_schema
    assert result_schema is not None
    assert not pydantic_agent._result_validators  # No custom validators
    assert pydantic_agent._current_result_retry == 0

    # Test function tools
    assert isinstance(pydantic_agent._function_tools, dict)

    # Test message history
    assert pydantic_agent.last_run_messages is None

    # Test researcher agent (minimal configuration)
    researcher = agents["researcher"]
    pydantic_agent = researcher.pydantic_agent
    assert isinstance(pydantic_agent.model, models.Model)
    assert pydantic_agent.model.name() == "openai:gpt-4"
    assert len(pydantic_agent._system_prompts) == 1
    assert pydantic_agent._default_retries == 1  # Default value
    assert pydantic_agent._max_result_retries == 1  # Default value


def test_create_agents_missing_response(runtime: RuntimeConfig):
    """Test agent creation with missing response definition."""
    config = {
        "responses": {},
        "agents": {
            "test": {
                "name": "Test Agent",
                "model": "openai:gpt-4",
                "result_model": "NonExistentResponse",
                "system_prompts": [],
            },
        },
    }
    with pytest.raises(exceptions.ConfigError) as exc_info:
        create_agents_from_config(AgentDefinition.model_validate(config), runtime)
    assert "NonExistentResponse" in str(exc_info.value)
