"""Test function-based prompt creation."""

from __future__ import annotations

import typing
from typing import Literal

import pytest

from llmling.prompts.function import create_prompt_from_callable
from llmling.prompts.models import ExtendedPromptArgument


def example_function(
    text: str,
    style: Literal["brief", "detailed"] = "brief",
    tags: list[str] | None = None,
) -> str:
    """Process text with given style and optional tags.

    Args:
        text: The input text to process
        style: Processing style (brief or detailed)
        tags: Optional tags to apply

    Returns:
        Processed text
    """
    return text


async def async_function(
    content: str,
    mode: str = "default",
) -> str:
    """Process content asynchronously.

    Args:
        content: Content to process
        mode: Processing mode

    Returns:
        Processed content
    """
    return content


def test_create_prompt_basic():
    """Test basic prompt creation from function."""
    prompt = create_prompt_from_callable(example_function)

    assert prompt.name == "example_function"
    assert "Process text with given style" in prompt.description
    assert len(prompt.arguments) == 3  # noqa: PLR2004
    assert len(prompt.messages) == 2  # noqa: PLR2004
    assert prompt.metadata["source"] == "function"
    assert "example_function" in prompt.metadata["import_path"]


def test_create_prompt_arguments():
    """Test argument conversion."""
    prompt = create_prompt_from_callable(example_function)
    args = {arg.name: arg for arg in prompt.arguments}

    # Check text argument
    assert isinstance(args["text"], ExtendedPromptArgument)
    assert args["text"].required is True
    assert args["text"].type_hint is str
    assert args["text"].description
    assert "input text to process" in args["text"].description.lower()

    # Check style argument - should be text type with enum values
    assert args["style"].required is False
    assert args["style"].type_hint is typing.Literal["brief", "detailed"]
    assert args["style"].default == "brief"
    assert "brief" in str(args["style"].description)
    assert "detailed" in str(args["style"].description)

    # Check tags argument
    assert args["tags"].required is False
    assert args["tags"].type_hint == (list[str] | None)
    assert args["tags"].default is None


def test_create_prompt_async():
    """Test prompt creation from async function."""
    prompt = create_prompt_from_callable(async_function)

    assert prompt.name == "async_function"
    assert "Process content asynchronously" in prompt.description
    assert len(prompt.arguments) == 2  # noqa: PLR2004

    args = {arg.name: arg for arg in prompt.arguments}
    description = args["content"].description
    assert description
    assert "Content to process" in description


def test_prompt_formatting():
    """Test that created prompts can be formatted."""
    prompt = create_prompt_from_callable(example_function)

    # Format with all arguments
    messages = prompt.format({
        "text": "sample",
        "style": "brief",
        "tags": ["test"],
    })
    formatted = messages[1].get_text_content()
    assert "text=sample" in formatted
    assert "style=brief" in formatted
    assert "tags=['test']" in formatted

    # Format with only required arguments
    messages = prompt.format({"text": "sample"})
    formatted = messages[1].get_text_content()
    assert "text=sample" in formatted
    assert "style=brief" in formatted  # Default value


def test_create_prompt_overrides():
    """Test prompt creation with overrides."""
    prompt = create_prompt_from_callable(
        example_function,
        name_override="custom_name",
        description_override="Custom description",
        template_override="Custom template: {text}",
    )

    assert prompt.name == "custom_name"
    assert prompt.description == "Custom description"

    # Test template override
    messages = prompt.format({"text": "test"})
    assert messages[1].get_text_content() == "Custom template: test"


def test_create_prompt_from_import_path():
    """Test prompt creation from import path."""
    prompt = create_prompt_from_callable("llmling.testing.processors.uppercase_text")

    assert prompt.name == "uppercase_text"
    assert "Convert text to uppercase" in prompt.description

    # Test formatting
    messages = prompt.format({"text": "test"})
    assert "text=test" in messages[1].get_text_content()


def test_create_prompt_invalid_import():
    """Test prompt creation with invalid import path."""
    with pytest.raises(ValueError, match="Could not import callable"):
        create_prompt_from_callable("nonexistent.module.function")


def test_argument_validation():
    """Test argument validation in created prompts."""
    prompt = create_prompt_from_callable(example_function)

    # Should fail without required argument
    with pytest.raises(ValueError, match="Missing required argument"):
        prompt.format({})

    # Should work with required argument
    messages = prompt.format({"text": "test"})
    assert len(messages) == 2  # noqa: PLR2004
    assert "text=test" in messages[1].get_text_content()


def test_system_message():
    """Test that system message contains function info."""
    prompt = create_prompt_from_callable(example_function)

    system_msg = prompt.messages[0]
    assert system_msg.role == "system"
    content = system_msg.get_text_content()
    assert "Function: example_function" in content
    assert "Description: Process text with given style" in content


def test_prompt_with_completions():
    """Test prompt creation with completion functions."""

    def get_language_completions(current: str) -> list[str]:
        languages = ["python", "javascript", "rust"]
        return [lang for lang in languages if lang.startswith(current)]

    def example_func(
        language: Literal["python", "javascript"],
        other: str,
    ) -> None:
        """Example function with completions."""

    prompt = create_prompt_from_callable(
        example_func, completions={"other": get_language_completions}
    )

    args = {arg.name: arg for arg in prompt.arguments}

    # Check literal type still works
    assert args["language"].completion_function is None

    # Check completion function
    assert args["other"].completion_function is not None
    assert args["other"].completion_function("py") == ["python"]
