from __future__ import annotations

import pytest

from llmling.core import exceptions
from llmling.prompts.models import ExtendedPromptArgument, PromptMessage, StaticPrompt


@pytest.mark.asyncio
async def test_render_prompt(runtime_config):
    """Test prompt rendering through runtime config."""
    msgs = [PromptMessage(role="user", content="Hello {name}")]
    args = [ExtendedPromptArgument(name="name", required=True)]
    prompt = StaticPrompt(
        name="test", description="Test prompt", messages=msgs, arguments=args
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
    msgs = [PromptMessage(role="user", content="Hello {name}")]
    args = [ExtendedPromptArgument(name="name", required=True)]
    prompt = StaticPrompt(
        name="test", description="Test prompt", messages=msgs, arguments=args
    )
    runtime_config._prompt_registry["test"] = prompt

    with pytest.raises(exceptions.LLMLingError, match="Missing required argument"):
        await runtime_config.render_prompt("test", {})
