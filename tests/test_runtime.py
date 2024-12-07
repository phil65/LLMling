from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from llmling.config.models import Config
from llmling.config.runtime import RuntimeConfig
from llmling.core import exceptions
from llmling.prompts.models import PromptMessage, PromptParameter, StaticPrompt


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_render_prompt(runtime_config):
    """Test prompt rendering through runtime config."""
    msgs = [PromptMessage(role="user", content="Hello {name}")]
    args = [PromptParameter(name="name", required=True)]
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
    args = [PromptParameter(name="name", required=True)]
    prompt = StaticPrompt(
        name="test", description="Test prompt", messages=msgs, arguments=args
    )
    runtime_config._prompt_registry["test"] = prompt

    with pytest.raises(exceptions.LLMLingError, match="Missing required argument"):
        await runtime_config.render_prompt("test", {})


@pytest.mark.asyncio
async def test_dynamic_prompt_arguments():
    """Test that DynamicPrompt properly extracts function arguments."""
    config = Config.model_validate({
        "prompts": {
            "test_prompt": {
                "type": "function",
                "import_path": "llmling.testing.processors.multiply",
                "description": "Test prompt",
            }
        }
    })

    async with RuntimeConfig.from_config(config) as runtime:
        prompt = runtime.get_prompt("test_prompt")

        # Verify argument extraction
        assert len(prompt.arguments) == 2  # noqa: PLR2004
        assert prompt.arguments[0].name == "text"
        assert prompt.arguments[0].required is True
        assert prompt.arguments[1].name == "times"
        assert prompt.arguments[1].required is False
        assert prompt.arguments[1].default == 2  # noqa: PLR2004

        # Verify formatting with both messages
        messages = await prompt.format({"text": "hello", "times": 3})
        assert len(messages) == 2  # noqa: PLR2004

        # Check system message
        assert messages[0].role == "system"
        assert messages[0].get_text_content() == "Content from test_prompt:"

        # Check user message with actual function result
        assert messages[1].role == "user"
        assert messages[1].get_text_content() == "hellohellohello"


@pytest.mark.asyncio
async def test_prompt_registration_methods(runtime_config: RuntimeConfig, tmp_path: Path):
    """Test all prompt registration methods."""
    # Register a static prompt
    runtime_config.register_static_prompt(
        name="greet",
        content="Hello {name}!",
        description="Simple greeting",
    )

    # Register a dynamic prompt using a function path
    runtime_config.register_dynamic_prompt(
        name="multiply",
        fn="llmling.testing.processors.multiply",
        description="Multiply text",
    )

    # Create prompt file in pytest-managed temporary directory
    prompt_file = tmp_path / "test_prompt.txt"
    prompt_file.write_text("Process this: {input}")

    # Register a file-based prompt
    runtime_config.register_file_prompt(
        name="process",
        path=prompt_file,
        description="Process input",
        output_format="text",
    )

    # Verify all prompts were registered
    prompts = runtime_config.get_prompts()
    assert len(prompts) == 3  # noqa: PLR2004

    # Test static prompt
    messages = await runtime_config.render_prompt("greet", {"name": "World"})
    assert messages[0].get_text_content() == "Hello World!"

    # Test dynamic prompt
    messages = await runtime_config.render_prompt("multiply", {"text": "abc", "times": 2})
    assert messages[1].get_text_content() == "abcabc"

    # Test file prompt
    messages = await runtime_config.render_prompt("process", {"input": "test data"})
    assert messages[0].get_text_content() == "Process this: test data"
