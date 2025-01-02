from __future__ import annotations

from pydantic import ValidationError
import pytest

from llmling.config.models import Config
from llmling.prompts.models import (
    PromptMessage,
    PromptParameter,
    StaticPrompt,
)


CONFIG_WITH_PROMPTS = """\
version: "1.0"
resources: {}
prompts:
  analyze:
    type: text
    name: analyze
    description: Analyze code
    messages:
      - role: user
        content: "Analyze this code: {code}"
    arguments:
      - name: code
        type: text
        description: Code to analyze
        required: true
"""

CONFIG_WITH_RESOURCE_PROMPTS = """
version: "1.0"
prompts:
  review:
    type: text
    name: review
    description: Review code and tests
    messages:
      - role: user
        content: [
                    {type: text, content: "Review this implementation:"},
                    {type: resource, content: "source://code.py", alt_text: "Source code"}
                ]
"""

INVALID_PROMPT_CONFIG = """\
version: "1.0"
prompts:
  invalid:
    type: text
    name: invalid
    messages:
      - role: invalid_role  # Invalid role
        content: test
"""


def test_config_with_prompts():
    """Test config with prompts section."""
    config = Config.from_yaml(CONFIG_WITH_PROMPTS)
    assert "analyze" in config.prompts
    assert config.prompts["analyze"].name == "analyze"


def test_config_with_resource_prompts():
    """Test config with prompts using resources."""
    config = Config.from_yaml(CONFIG_WITH_RESOURCE_PROMPTS)
    assert config.prompts["review"]
    msg = config.prompts["review"].messages[0]
    assert isinstance(msg.content, list)
    assert len(msg.content) == 2  # noqa: PLR2004
    assert msg.content[0].type == "text"
    assert msg.content[1].type == "resource"
    assert msg.content[1].content == "source://code.py"


def test_invalid_prompt_config():
    """Test invalid prompt configurations."""
    with pytest.raises(ValidationError):
        Config.from_yaml(INVALID_PROMPT_CONFIG)


@pytest.mark.asyncio
async def test_prompt_format():
    """Test prompt message formatting."""
    prompt = StaticPrompt(
        name="test",
        description="Test prompt",
        messages=[
            PromptMessage(role="user", content="Hello {name}"),
            PromptMessage(role="user", content="Age: {age}"),
        ],
        arguments=[
            PromptParameter(name="name", required=True),
            PromptParameter(name="age", required=False),
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
    prompt = StaticPrompt(
        name="test",
        description="Test prompt",
        messages=[PromptMessage(role="user", content="Test {required_arg}")],
        arguments=[PromptParameter(name="required_arg", required=True)],
    )

    # Should raise when missing required argument
    with pytest.raises(ValueError, match="Missing required argument"):
        await prompt.format({})

    # Should work with required argument
    messages = await prompt.format({"required_arg": "value"})
    assert messages[0].get_text_content() == "Test value"
