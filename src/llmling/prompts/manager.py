"""Prompt management and composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling.core.log import get_logger
from llmling.llm.base import Message
from llmling.prompts.models import MessageContext, PromptPriority, SystemPrompt


if TYPE_CHECKING:
    from llmling.tools.base import LLMCallableTool

logger = get_logger(__name__)


class PromptManager:
    """Manages prompt composition and organization."""

    def create_messages(
        self,
        context: MessageContext,
        *,
        tools: list[LLMCallableTool] | None = None,
    ) -> list[Message]:
        """Create messages from context.

        Args:
            context: Message construction context
            tools: Optional list of tools being used

        Returns:
            List of messages for LLM
        """
        messages: list[Message] = []

        # Collect all system prompts
        prompts = list(context.system_prompts)  # Copy original prompts

        # Add tool prompts if any tools provided
        if tools:
            prompts.extend(self._get_tool_prompts(tools))

        # Sort by priority and add to messages
        if prompts:
            msgs = self._create_system_messages(sorted(prompts, key=lambda p: p.priority))
            messages.extend(msgs)

        # Add user content
        if context.user_content or context.content_items:
            msg = Message(
                role="user",
                content=context.user_content,
                content_items=context.content_items,
            )
            messages.append(msg)

        return messages

    def _get_tool_prompts(
        self,
        tools: list[LLMCallableTool],
    ) -> list[SystemPrompt]:
        """Get system prompts from tools."""
        return [
            SystemPrompt(
                content=tool.system_prompt,
                source=f"tool:{tool.name}",
                priority=PromptPriority.TOOL,
                metadata={"tool_name": tool.name},
            )
            for tool in tools
            if hasattr(tool, "system_prompt")
        ]

    def _create_system_messages(
        self,
        prompts: list[SystemPrompt],
    ) -> list[Message]:
        """Create system messages from prompts."""
        return [
            Message(
                role="system",
                content=p.content,
                metadata={"source": p.source} if p.source else None,
            )
            for p in prompts
        ]
