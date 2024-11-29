"""Prompt management and composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling.core.log import get_logger
from llmling.core.typedefs import Message, MessageContent


if TYPE_CHECKING:
    from llmling.tools.base import LLMCallableTool


logger = get_logger(__name__)


class PromptManager:
    """Manages prompt composition and system prompts."""

    def create_messages(
        self,
        *,
        system_content: str | None = None,
        user_content: str | None = None,
        content_items: list[MessageContent] | None = None,
        tools: list[LLMCallableTool] | None = None,
    ) -> list[Message]:
        """Create message list for LLM interaction.

        Args:
            system_content: Optional system message content
            user_content: Optional user message content
            content_items: Optional list of content items
            tools: Optional list of tools being used

        Returns:
            List of messages for LLM
        """
        messages: list[Message] = []

        # Add system message if provided
        if system_content:
            messages.append(
                Message(
                    role="system",
                    content=system_content,
                )
            )

        # Add tool system prompts if any
        if tools:
            messages.extend(
                Message(
                    role="system",
                    content=tool.system_prompt,
                    name=tool.name,
                )
                for tool in tools
                if tool.system_prompt
            )

        # Add user content
        if user_content or content_items:
            messages.append(
                Message(
                    role="user",
                    content=user_content or "",
                    content_items=content_items or [],
                )
            )

        return messages
