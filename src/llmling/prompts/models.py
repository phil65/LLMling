"""Prompt-related models."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llmling.llm.base import MessageContent  # noqa: TC001


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
