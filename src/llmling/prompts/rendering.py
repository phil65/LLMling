"""Prompt rendering utilities."""

from __future__ import annotations

from typing import Any

from llmling.core import exceptions
from llmling.prompts.models import Prompt, PromptMessage, PromptResult


async def render_prompt(
    prompt: Prompt,
    arguments: dict[str, Any],
) -> PromptResult:
    """Render a prompt template with arguments."""
    try:
        # Validate arguments
        prompt.validate_arguments(arguments)

        # Render each message
        rendered_messages = []
        for message in prompt.messages:
            content = message.content.format(**arguments)
            rendered_messages.append(
                PromptMessage(
                    role=message.role,
                    content=content,
                    name=message.name,
                )
            )

        return PromptResult(
            messages=rendered_messages,
            metadata={
                "prompt_name": prompt.name,
                "arguments": arguments,
            },
        )
    except KeyError as exc:
        msg = f"Missing argument in template: {exc}"
        raise exceptions.LLMLingError(msg) from exc
    except ValueError as exc:
        msg = f"Invalid arguments: {exc}"
        raise exceptions.LLMLingError(msg) from exc
