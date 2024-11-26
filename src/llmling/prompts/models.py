"""Prompt-related models."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from enum import Enum, IntEnum
from typing import Any, Literal

import mcp
from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmling.core.typedefs import MessageContent
from llmling.resources.models import LoadedResource  # noqa: TC001


MessageRole = Literal["system", "user", "assistant", "tool"]


class ArgumentType(str, Enum):
    """Types of prompt arguments that support completion."""

    TEXT = "text"
    FILE = "file"
    ENUM = "enum"
    RESOURCE = "resource"
    TOOL = "tool"


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


class ExtendedPromptArgument(BaseModel):
    """Extended argument definition with completion support.

    This extends the base MCP PromptArgument with additional fields
    for completion and validation.
    """

    name: str
    description: str | None = None
    required: bool | None = None
    # Extended fields for completion support
    type: ArgumentType = ArgumentType.TEXT
    enum_values: list[str] | None = None  # For enum type
    file_patterns: list[str] | None = None  # For file type
    resource_types: list[str] | None = None  # For resource type
    tool_names: list[str] | None = None  # For tool type
    default: Any | None = None

    model_config = ConfigDict(extra="allow")

    def to_mcp_argument(self) -> mcp.types.PromptArgument:
        """Convert to MCP PromptArgument."""
        return mcp.types.PromptArgument(
            name=self.name,
            description=self.description,
            required=self.required,
        )

    @model_validator(mode="after")
    def validate_type_specific_fields(self) -> ExtendedPromptArgument:
        """Validate fields specific to argument types."""
        match self.type:
            case ArgumentType.ENUM:
                if not self.enum_values:
                    msg = "enum_values required for enum type"
                    raise ValueError(msg)
            case ArgumentType.FILE:
                if not self.file_patterns:
                    self.file_patterns = ["*"]  # Default to all files
            case ArgumentType.RESOURCE:
                if not self.resource_types:
                    self.resource_types = ["*"]  # Default to all resources
            case ArgumentType.TOOL:
                if not self.tool_names:
                    self.tool_names = ["*"]  # Default to all tools
        return self


class ResolvedContent(BaseModel):
    """Content with resolved resources."""

    original: MessageContent
    resolved: LoadedResource | None = None
    resolved_at: datetime | None = None

    model_config = ConfigDict(frozen=True)


class PromptMessage(BaseModel):
    """A message in a prompt."""

    role: MessageRole
    content: str | MessageContent | list[MessageContent] = ""
    resolved_content: list[ResolvedContent] | None = None

    model_config = ConfigDict(frozen=True)

    def needs_resolution(self) -> bool:
        """Check if message has unresolved resources."""
        contents = self.get_content_items()
        return any(item.type == "resource" for item in contents)

    def get_content_items(self) -> list[MessageContent]:
        """Get all content items."""
        match self.content:
            case str():
                return [MessageContent.text(self.content)]
            case MessageContent():
                return [self.content]
            case list():
                return self.content
            case _:
                return [MessageContent.text(str(self.content))]

    @model_validator(mode="before")
    @classmethod
    def ensure_content_items(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Ensure backwards compatibility for content field."""
        if isinstance(data, dict):
            content = data.get("content", "")
            match content:
                case str():
                    # Convert string to text content
                    data["content"] = MessageContent.text(content)
                case MessageContent():
                    # Already correct format
                    pass
                case list():
                    # Ensure all items are MessageContent
                    data["content"] = [
                        item
                        if isinstance(item, MessageContent)
                        else MessageContent.text(str(item))
                        for item in content
                    ]
                case _:
                    # Convert anything else to string
                    data["content"] = MessageContent.text(str(content))
        return data

    def get_text_content(self) -> str:
        """Get text content for backwards compatibility."""
        match self.content:
            case str():
                return self.content
            case MessageContent() if self.content.type == "text":
                return self.content.content
            case list() if self.content:
                # Get first text content or first content
                text_items = [
                    item.content for item in self.content if item.type == "text"
                ]
                return text_items[0] if text_items else self.content[0].content
            case _:
                return ""


class Prompt(BaseModel):
    """Prompt template definition."""

    name: str
    description: str
    messages: list[PromptMessage]
    arguments: list[ExtendedPromptArgument] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    def to_mcp_prompt(self) -> mcp.types.Prompt:
        """Convert to MCP Prompt."""
        return mcp.types.Prompt(
            name=self.name,
            description=self.description,
            arguments=[arg.to_mcp_argument() for arg in self.arguments],
        )

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
    resolved_at: datetime | None = None

    model_config = ConfigDict(frozen=True)
