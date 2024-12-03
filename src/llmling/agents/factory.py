"""Factory for creating Pydantic-AI agents from configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, create_model

from llmling.agents.agent import LLMlingAgent
from llmling.agents.models import (
    AgentDefinition,
    ResponseDefinition,
    SystemPrompt,
)
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.utils import importing


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling.agents.models import (
        AgentConfig,
        AgentDefinition,
        ResponseDefinition,
        SystemPrompt,
    )
    from llmling.config.runtime import RuntimeConfig


logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


def create_agents_from_config(
    config: AgentDefinition,
    runtime: RuntimeConfig,
) -> dict[str, LLMlingAgent[Any]]:
    """Create all agents from configuration.

    Args:
        config: Complete agent configuration
        runtime: Runtime configuration

    Returns:
        Dictionary mapping agent IDs to LLMling agents

    Raises:
        ConfigError: If configuration is invalid
    """
    agents = {}
    for agent_id, agent_config in config.agents.items():
        try:
            agents[agent_id] = _create_single_agent(
                agent_config,
                config.responses,
                runtime,
            )
        except Exception as exc:
            msg = f"Failed to create agent {agent_id}: {exc}"
            raise exceptions.ConfigError(msg) from exc
    return agents


def _create_single_agent(
    agent_config: AgentConfig,
    responses: dict[str, ResponseDefinition],
    runtime: RuntimeConfig,
) -> LLMlingAgent[Any]:
    """Internal helper to create a single agent."""
    result_def = responses[agent_config.result_model]
    result_model = _create_response_model(
        agent_config.result_model,
        result_def,
    )

    return LLMlingAgent(
        runtime=runtime,
        result_type=result_model,
        model=agent_config.model,
        system_prompt=_create_system_prompts(agent_config.system_prompts),
        name=agent_config.name,
        **agent_config.model_settings,
    )


def _create_response_model(name: str, definition: ResponseDefinition) -> type[BaseModel]:
    """Create a Pydantic model from response definition."""
    fields: dict[str, Any] = {}  # Change type hint

    for field_name, field_def in definition.fields.items():
        # Convert string type to actual type
        type_hint = _parse_type_annotation(field_def.type)

        # Create field with metadata
        field = Field(
            description=field_def.description,
            **(field_def.constraints or {}),
        )
        fields[field_name] = (type_hint, field)

    return create_model(
        name,
        __config__=ConfigDict(frozen=True),
        __doc__=definition.description,
        **fields,
    )


def _create_system_prompts(prompts: list[SystemPrompt]) -> Sequence[str]:
    """Convert system prompt configs to actual prompts."""
    result: list[str] = []

    for prompt in prompts:
        match prompt.type:
            case "text":
                result.append(prompt.value)
            case "function":
                func = importing.import_callable(prompt.value)
                # Need to ensure function returns str
                result.append(str(func()))
            case "template":
                result.append(prompt.value)

    return result


def _parse_type_annotation(type_str: str) -> Any:
    """Convert string type annotation to actual type."""
    import typing

    # Handle basic types
    basic_types = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "dict": dict,
        "list": list,
    }

    if type_str in basic_types:
        return basic_types[type_str]

    # Handle generic types (e.g. list[str])
    if "[" in type_str:
        container, inner = type_str.split("[", 1)
        inner = inner.rstrip("]")

        container_type = getattr(typing, container.capitalize())
        inner_type = _parse_type_annotation(inner)

        return container_type[inner_type]

    msg = f"Unknown type: {type_str}"
    raise ValueError(msg)
