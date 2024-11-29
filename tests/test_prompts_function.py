from __future__ import annotations

from typing import Literal

import pytest

from llmling.prompts.function import create_prompt_from_callable
from llmling.prompts.models import ArgumentType, ExtendedPromptArgument


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
    assert args["text"].type == ArgumentType.TEXT
    assert args["text"].description
    assert "input text to process" in args["text"].description.lower()

    # Check style argument
    assert args["style"].required is False
    assert args["style"].type == ArgumentType.ENUM
    assert args["style"].enum_values == ["brief", "detailed"]
    assert args["style"].default == "brief"

    # Check tags argument
    assert args["tags"].required is False
    assert args["tags"].type == ArgumentType.TEXT  # Changed from ENUM
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
    assert prompt.messages[1].content
    assert "Custom template" in prompt.messages[1].get_text_content()


def test_create_prompt_from_import_path():
    """Test prompt creation from import path."""
    prompt = create_prompt_from_callable("llmling.testing.processors.uppercase_text")

    assert prompt.name == "uppercase_text"
    assert "Convert text to uppercase" in prompt.description


def test_create_prompt_invalid_import():
    """Test prompt creation with invalid import path."""
    with pytest.raises(ValueError, match="Could not import callable"):
        create_prompt_from_callable("nonexistent.module.function")


def test_argument_types():
    """Test various argument type conversions."""

    def func_with_types(
        text: str,
        count: int,
        flag: bool,
        items: list[str],
        choice: Literal["a", "b"],
    ) -> None:
        """Test function with various types."""

    prompt = create_prompt_from_callable(func_with_types)
    args = {arg.name: arg for arg in prompt.arguments}

    assert args["text"].type == ArgumentType.TEXT
    assert args["count"].type == ArgumentType.TEXT
    assert args["flag"].type == ArgumentType.ENUM
    assert args["flag"].enum_values == ["true", "false"]
    assert args["items"].type == ArgumentType.TEXT
    assert args["choice"].type == ArgumentType.ENUM
    assert args["choice"].enum_values == ["a", "b"]
