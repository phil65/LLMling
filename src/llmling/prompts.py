"""Message content types for prompts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel, Field


class PromptType(str, Enum):
    SYSTEM = "system"
    """System-level prompts that set behavior and context"""

    USER = "user"
    """User input prompts"""

    ASSISTANT = "assistant"
    """Assistant responses in conversation history"""

    FUNCTION = "function"
    """Function calls and their responses"""

    IMAGE = "image"
    """Image URL prompts for multimodal models"""



class BasePrompt(ABC, BaseModel):
    """Base class for all prompt types."""
    order: int = Field(ge=0)

    @property
    @abstractmethod
    def prompt_type(self) -> PromptType:
        """Get the type of this prompt."""

    @abstractmethod
    def to_message_content(self) -> dict:
        """Convert to complete message dictionary for LLM API."""


class TextContent(BasePrompt):
    """Text-based prompt content."""
    text: str
    type: PromptType = Field(..., exclude=False)

    @property
    def prompt_type(self) -> PromptType:
        return self.type

    def to_message_content(self) -> dict:
        return {
            "role": self.type,
            "content": self.text
        }


class ImageContent(BasePrompt):
    """Image-based prompt content."""
    url: str

    @property
    def prompt_type(self) -> PromptType:
        return PromptType.IMAGE

    def to_message_content(self) -> dict:
        return {
            "role": "user",
            "content": [{"type": "image_url", "image_url": self.url}]
        }



__all__ = [
    "PromptType",
    "BasePrompt",
    "TextContent",
    "ImageContent",
]


if __name__ == "__main__":
    # Example system prompt
    system = TextContent(
        type=PromptType.SYSTEM,
        text="You are a helpful assistant.",
        order=0
    )

    # Example user prompt
    user = TextContent(
        type=PromptType.USER,
        text="What is the weather like?",
        order=1
    )

    # Example image prompt
    image = ImageContent(
        url="https://www.dhs.wisconsin.gov/sites/default/files/styles/large/public/dam/image/5/thermometer-inthe-snow.jpg?itok=rWkiCLO_",
        order=2
    )

    # Print converted message formats
    prompts = [system, user, image]
    for p in prompts:
        print(f"\n{p.prompt_type}:")
        print(p.to_message_content())
