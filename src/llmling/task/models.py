"""Task execution models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from llmling.config import Context, TaskSettings
from llmling.processors.registry import ProcessingStep


class TaskContext(BaseModel):
    """Context configuration for a task."""

    context: Context
    processors: list[ProcessingStep]
    inherit_tools: bool = True

    model_config = ConfigDict(frozen=True)


class TaskProvider(BaseModel):
    """Provider configuration for a task."""

    name: str
    model: str
    settings: TaskSettings | None = None

    model_config = ConfigDict(frozen=True)


class TaskResult(BaseModel):
    """Result of a task execution."""

    content: str
    model: str
    context_metadata: dict[str, Any]
    completion_metadata: dict[str, Any]

    model_config = ConfigDict(frozen=True)
