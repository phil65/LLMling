from __future__ import annotations

import pytest

from llmling.config.models import TextResource
from llmling.core import exceptions
from llmling.prompts.models import ExtendedPromptArgument, PromptMessage, StaticPrompt
from llmling.resources.loaders.text import TextResourceLoader


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


@pytest.mark.asyncio
async def test_resource_uri_after_registration(runtime_config):
    """Test that resources get URIs when registered."""
    runtime_config._resource_registry.loader_registry["text"] = TextResourceLoader

    # Create a resource without URI
    resource = TextResource(content="test content")
    assert resource.uri is None

    # Register it
    runtime_config.register_resource("test", resource)

    # Get it back from registry and check URI
    registered = runtime_config._resource_registry["test"]
    assert registered.uri == "text://test"  # Should have canonical URI

    # Check that original resource wasn't modified
    assert resource.uri is None

    # Check that URI is preserved when getting through public interface
    assert runtime_config.get_resource("test").uri == "text://test"
