"""Prompt-related models."""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from llmling.llm.base import MessageContent  # noqa: TC001


MessageRole = Literal["system", "user", "assistant", "tool"]


class PromptPriority(IntEnum):
    """Priority levels for system prompts."""

    TOOL = 100  # Tool-specific instructions
    SYSTEM = 200  # User-provided system prompts
    OVERRIDE = 300  # High-priority overrides


class SystemPrompt(BaseModel):
    """System prompt configuration."""

    content: str
    source: str = ""  # e.g., "tool:browser", "user", "config"
    priority: PromptPriority = PromptPriority.SYSTEM
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class MessageContext(BaseModel):
    """Context for message construction."""

    system_prompts: list[SystemPrompt] = Field(default_factory=list)
    user_content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_items: list[MessageContent] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class PromptArgument(BaseModel):
    """Argument definition for prompt templates."""

    name: str
    description: str
    required: bool = True
    type: str = "string"
    default: Any | None = None

    model_config = ConfigDict(frozen=True)


class PromptMessage(BaseModel):
    """Single message in a prompt template."""

    role: MessageRole
    content: str
    name: str | None = None

    model_config = ConfigDict(frozen=True)


class Prompt(BaseModel):
    """Prompt template definition."""

    name: str
    description: str
    messages: list[PromptMessage]
    arguments: list[PromptArgument] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    def validate_arguments(self, provided: dict[str, Any]) -> None:
        """Validate provided arguments against requirements."""
        required = {arg.name for arg in self.arguments if arg.required}
        missing = required - set(provided)
        if missing:
            msg = f"Missing required arguments: {', '.join(missing)}"
            raise ValueError(msg)


class PromptResult(BaseModel):
    """Result of rendering a prompt template."""

    messages: list[PromptMessage]
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)
