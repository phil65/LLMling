"""Models for agent configuration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import models  # noqa: TC002


class ResponseField(BaseModel):
    """Field definition for agent responses."""

    type: str
    description: str | None = None
    constraints: dict[str, Any] | None = None


class ResponseDefinition(BaseModel):
    """Definition of an agent response type."""

    description: str | None = None
    fields: dict[str, ResponseField]


class SystemPrompt(BaseModel):
    """System prompt configuration."""

    type: Literal["text", "function", "template"]
    value: str


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    name: str
    description: str | None = None
    model: models.Model | models.KnownModelName | None = None
    model_settings: dict[str, Any] = Field(default_factory=dict)
    result_model: str
    deps_type: str | None = None
    system_prompts: list[SystemPrompt]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentDefinition(BaseModel):
    """Complete agent definition including responses."""

    responses: dict[str, ResponseDefinition]
    agents: dict[str, AgentConfig]

    model_config = ConfigDict(arbitrary_types_allowed=True)


# AgentDefinition.model_rebuild()
