"""Common type definitions for llmling."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator
from schemez import Schema


ProcessorCallable = Callable[[str, Any], str | Awaitable[str]]
MetadataDict = dict[str, Any]

MessageContentType = Literal["text", "resource", "image_url", "image_base64"]

# Our internal role type (could include more roles)
MessageRole = Literal["system", "user", "assistant", "tool"]


class ProcessingStep(Schema):  # type: ignore[no-redef]
    """Configuration for a processing step."""

    name: str
    parallel: bool = False
    required: bool = True
    kwargs: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class MessageContent(Schema):
    """Content item in a message."""

    type: MessageContentType
    content: str  # The actual content (text/uri/url/base64)
    alt_text: str | None = None  # For images or resource descriptions

    model_config = ConfigDict(frozen=True)

    @classmethod
    def text(cls, content: str) -> MessageContent:
        """Create text content."""
        return cls(type="text", content=content)

    @classmethod
    def resource(cls, uri: str, description: str | None = None) -> MessageContent:
        """Create resource reference."""
        return cls(type="resource", content=uri, alt_text=description)

    @classmethod
    def image(
        cls,
        url_or_data: str,
        *,
        is_base64: bool = False,
        alt_text: str | None = None,
    ) -> MessageContent:
        """Create image content."""
        type_ = "image_base64" if is_base64 else "image_url"
        return cls(type=type_, content=url_or_data, alt_text=alt_text)


class ToolCall(Schema):
    """A tool call request from the LLM."""

    id: str  # Required by OpenAI
    name: str
    parameters: dict[str, Any]

    model_config = ConfigDict(frozen=True)


class Message(Schema):
    """A chat message."""

    role: MessageRole
    content: str = ""  # Keep for backward compatibility
    content_items: list[MessageContent] = Field(default_factory=list)
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="before")
    @classmethod
    def ensure_content_items(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Ensure content_items is populated from content if empty."""
        if isinstance(data, dict):
            content = data.get("content", "")
            content_items = data.get("content_items", [])
            # Only create content_items from content if we have content and no items
            if content and not content_items:
                data["content_items"] = [
                    MessageContent(type="text", content=content).model_dump()
                ]
            # Always keep content in sync with first text content item
            elif content_items:
                text_items = [
                    item
                    for item in content_items
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                if text_items:
                    data["content"] = text_items[0]["content"]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Create message from dict representation."""
        # Handle content_items if present
        if "content_items" in data and isinstance(data["content_items"], list):
            data["content_items"] = [
                MessageContent(**item) if isinstance(item, dict) else item
                for item in data["content_items"]
            ]
        # Handle tool_calls if present
        if "tool_calls" in data and isinstance(data["tool_calls"], list):
            data["tool_calls"] = [
                ToolCall(**call) if isinstance(call, dict) else call
                for call in data["tool_calls"]
            ]
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return self.model_dump(exclude_none=True)

    @property
    def has_tools(self) -> bool:
        """Check if message contains tool calls."""
        return bool(self.tool_calls)

    @property
    def has_images(self) -> bool:
        """Check if message contains images."""
        return any(
            item.type in ("image_url", "image_base64") for item in self.content_items
        )
