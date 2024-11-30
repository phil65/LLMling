from __future__ import annotations

from pydantic import ValidationError
import pytest

from llmling.config.models import Config
from llmling.prompts.models import (
    ExtendedPromptArgument,
    MessageContent,
    Prompt,
    PromptMessage,
)


def test_config_with_prompts():
    """Test config with prompts section."""
    config_data = {
        "version": "1.0",
        "resources": {},
        "prompts": {
            "analyze": {
                "name": "analyze",
                "description": "Analyze code",
                "messages": [{"role": "user", "content": "Analyze this code: {code}"}],
                "arguments": [
                    {
                        "name": "code",
                        "type": "text",
                        "description": "Code to analyze",
                        "required": True,
                    }
                ],
            }
        },
    }
    config = Config.model_validate(config_data)
    assert "analyze" in config.prompts
    assert config.prompts["analyze"].name == "analyze"


def test_config_with_resource_prompts():
    """Test config with prompts using resources."""
    config_data = {
        "version": "1.0",
        "resources": {},
        "prompts": {
            "review": {
                "name": "review",
                "description": "Review code and tests",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            MessageContent(
                                type="text", content="Review this implementation:"
                            ),
                            MessageContent(
                                type="resource",
                                content="source://code.py",
                                alt_text="Source code",
                            ),
                        ],
                    }
                ],
            }
        },
    }
    config = Config.model_validate(config_data)
    assert config.prompts["review"]
    msg = config.prompts["review"].messages[0]  # type: ignore
    assert isinstance(msg.content, list)
    assert len(msg.content) == 2  # noqa: PLR2004
    assert msg.content[0].type == "text"
    assert msg.content[1].type == "resource"
    assert msg.content[1].content == "source://code.py"


def test_invalid_prompt_config():
    """Test invalid prompt configurations."""
    with pytest.raises(ValidationError):
        Config.model_validate({
            "version": "1.0",
            "prompts": {
                "invalid": {
                    "name": "invalid",
                    "messages": [
                        {
                            "role": "invalid_role",  # Invalid role
                            "content": "test",
                        }
                    ],
                }
            },
        })


@pytest.fixture
def sample_prompt() -> Prompt:
    """Create a sample prompt for testing."""
    arg = ExtendedPromptArgument(
        name="name",
        type="text",
        description="Name to greet",
        required=True,
    )
    return Prompt(
        name="test",
        description="Test prompt",
        messages=[PromptMessage(role="user", content="Hello {name}")],
        arguments=[arg],
    )


@pytest.mark.asyncio
async def test_prompt_format():
    """Test prompt message formatting."""
    prompt = Prompt(
        name="test",
        description="Test prompt",
        messages=[
            PromptMessage(role="user", content="Hello {name}"),
            PromptMessage(role="user", content="Age: {age}"),
        ],
        arguments=[
            ExtendedPromptArgument(name="name", required=True),
            ExtendedPromptArgument(name="age", required=False),
        ],
    )

    # Test with all arguments
    messages = await prompt.format({"name": "Alice", "age": "30"})
    assert len(messages) == 2  # noqa: PLR2004
    assert messages[0].get_text_content() == "Hello Alice"
    assert messages[1].get_text_content() == "Age: 30"

    # Test with only required arguments
    messages = await prompt.format({"name": "Bob"})
    assert messages[0].get_text_content() == "Hello Bob"
    assert messages[1].get_text_content() == "Age: "


@pytest.mark.asyncio
async def test_prompt_validation():
    """Test prompt argument validation."""
    prompt = Prompt(
        name="test",
        description="Test prompt",
        messages=[PromptMessage(role="user", content="Test {required_arg}")],
        arguments=[ExtendedPromptArgument(name="required_arg", required=True)],
    )

    # Should raise when missing required argument
    with pytest.raises(ValueError, match="Missing required argument"):
        await prompt.format({})

    # Should work with required argument
    messages = await prompt.format({"required_arg": "value"})
    assert messages[0].get_text_content() == "Test value"
