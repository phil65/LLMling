"""Task execution models."""

from __future__ import annotations  # noqa: I001

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from llmling.core.typedefs import ProcessingStep  # noqa: TC001
from llmling.config import Context, TaskSettings  # noqa: TC001


class TaskContext(BaseModel):
    """Context configuration for a task.

    Contains all necessary information about the task's context, including:
    - Base context configuration from YAML
    - Runtime context variables passed during execution
    - Processing steps to apply
    - Tool configuration
    """

    context: Context
    processors: list[ProcessingStep] = Field(default_factory=list)
    inherit_tools: bool = False
    tools: list[str] | None = None
    runtime_context: dict[str, Any] | None = None
    tool_choice: Literal["none", "auto"] | str | None = None  # noqa: PYI051

    model_config = ConfigDict(frozen=True)


class TaskProvider(BaseModel):
    """Provider configuration for a task.

    Contains all necessary information about the LLM provider, including:
    - Provider name for registry lookup
    - Model identifier
    - Display name for logging/UI
    - Task-specific settings that override provider defaults
    """

    name: str  # Provider lookup key
    display_name: str = ""  # Human readable name
    model: str
    settings: TaskSettings | None = None

    model_config = ConfigDict(frozen=True)


class TaskResult(BaseModel):
    """Result of a task execution.

    Contains:
    - Generated content from the LLM
    - Model identifier used for generation
    - Metadata from context processing
    - Metadata from LLM completion
    """

    content: str
    model: str
    context_metadata: dict[str, Any]
    completion_metadata: dict[str, Any]

    model_config = ConfigDict(frozen=True)
