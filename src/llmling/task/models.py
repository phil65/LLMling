"""Task execution models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from llmling.config import Context, TaskSettings  # noqa: TCH001
from llmling.core.typedefs import Content, ContentData, ContentType, ProcessingStep


class TaskContext(BaseModel):
    """Context configuration for a task."""

    context: Context
    processors: list[ProcessingStep]
    inherit_tools: bool = False  # Set default value to False
    tools: list[str] | None = None
    tool_choice: Literal["none", "auto"] | str | None = None  # noqa: PYI051

    model_config = ConfigDict(frozen=True)


class TaskProvider(BaseModel):
    """Provider configuration for a task."""

    name: str  # Provider lookup key
    display_name: str = ""  # Human readable name
    model: str
    settings: TaskSettings | None = None

    model_config = ConfigDict(frozen=True)


class TaskResult(BaseModel):
    """Result of a task execution."""

    content: Content[ContentData]
    model: str
    context_metadata: dict[str, Any]
    completion_metadata: dict[str, Any]
    is_chunk: bool = False  # Indicate if this is a streaming chunk

    model_config = ConfigDict(frozen=True)

    @classmethod
    def create_chunk(
        cls, content: str | Content[ContentData], model: str, **kwargs: Any
    ) -> TaskResult:
        """Create a streaming chunk result."""
        if isinstance(content, str):
            content = Content(
                type=ContentType.TEXT, data=content, metadata={"chunk": True}
            )
        return cls(
            content=content,
            model=model,
            is_chunk=True,
            context_metadata=kwargs.get("context_metadata", {}),
            completion_metadata=kwargs.get("completion_metadata", {}),
        )

    def get_text(self) -> str:
        """Get text content with safety checks."""
        if isinstance(self.content, Content):
            if self.content.type == ContentType.TEXT:
                return str(self.content.data)
            return f"[{self.content.type.value} content]"
        return str(self.content)
