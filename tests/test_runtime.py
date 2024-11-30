from __future__ import annotations

import pytest

from llmling.core import exceptions
from llmling.prompts.models import ExtendedPromptArgument, Prompt, PromptMessage


@pytest.mark.asyncio
async def test_render_prompt(runtime_config):
    """Test prompt rendering through runtime config."""
    prompt = Prompt(
        name="test",
        description="Test prompt",
        messages=[PromptMessage(role="user", content="Hello {name}")],
        arguments=[ExtendedPromptArgument(name="name", required=True)],
    )
    runtime_config._prompt_registry["test"] = prompt

    messages = await runtime_config.render_prompt("test", {"name": "World"})
    assert len(messages) == 1
    assert messages[0].get_text_content() == "Hello World"


async def test_render_prompt_not_found(runtime_config):
    """Test error handling for non-existent prompts."""
    with pytest.raises(exceptions.LLMLingError, match="Item not found"):
        await runtime_config.render_prompt("nonexistent")


@pytest.mark.asyncio
async def test_render_prompt_validation_error(runtime_config):
    """Test error handling for invalid arguments."""
    prompt = Prompt(
        name="test",
        description="Test prompt",
        messages=[PromptMessage(role="user", content="Hello {name}")],
        arguments=[ExtendedPromptArgument(name="name", required=True)],
    )
    runtime_config._prompt_registry["test"] = prompt

    with pytest.raises(exceptions.LLMLingError, match="Missing required argument"):
        await runtime_config.render_prompt("test", {})
