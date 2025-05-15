"""Resource models."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field, model_validator
from schemez import Schema

from llmling.core.typedefs import MessageContent


class ProcessingContext(Schema):  # type: ignore[no-redef]
    """Context for processor execution."""

    original_content: str
    current_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    kwargs: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class ResourceMetadata(Schema):
    """MCP resource metadata."""

    uri: str
    mime_type: str
    name: str | None = None
    description: str | None = None
    size: int | None = None
    modified: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class LoadedResource(Schema):
    """Result of loading and processing a context."""

    content: str = ""  # Keep for backward compatibility
    content_items: list[MessageContent] = Field(default_factory=list)
    source_type: str | None = None
    metadata: ResourceMetadata
    etag: str | None = None  # For MCP caching support

    @model_validator(mode="before")
    @classmethod
    def ensure_content_sync(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Ensure content and content_items are in sync."""
        if isinstance(data, dict):
            content = data.get("content", "")
            content_items = data.get("content_items", [])

            # If we have content but no items, create a text item
            if content and not content_items:
                msg_content = MessageContent(type="text", content=content)
                data["content_items"] = [msg_content.model_dump()]
            # If we have items but no content, use first text item's content
            elif content_items and not content:
                text_items = [
                    i
                    for i in content_items
                    if isinstance(i, dict) and i.get("type") == "text"
                ]
                if text_items:
                    data["content"] = text_items[0]["content"]
        return data
