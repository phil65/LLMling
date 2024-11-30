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
    result = text
    if style == "detailed":
        result = f"{result} (detailed)"
    if tags:
        result = f"{result} [tags: {', '.join(tags)}]"
    return result


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
    if mode == "upper":
        return content.upper()
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

    # Check style argument
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


@pytest.mark.asyncio
async def test_prompt_formatting():
    """Test that created prompts format with function results."""
    prompt = create_prompt_from_callable(example_function)

    # Format with all arguments
    messages = await prompt.format({
        "text": "sample",
        "style": "detailed",
        "tags": ["test"],
    })
    result = messages[1].get_text_content()
    assert result == "sample (detailed) [tags: test]"

    # Format with only required arguments
    messages = await prompt.format({"text": "sample"})
    assert messages[1].get_text_content() == "sample"  # Uses default brief style


def test_create_prompt_overrides():
    """Test prompt creation with overrides."""
    prompt = create_prompt_from_callable(
        example_function,
        name_override="custom_name",
        description_override="Custom description",
        template_override="Result: {result}",
    )

    assert prompt.name == "custom_name"
    assert prompt.description == "Custom description"
    assert prompt.messages[1].content.content == "Result: {result}"  # type: ignore


@pytest.mark.asyncio
async def test_create_prompt_from_import_path():
    """Test prompt creation from import path."""
    prompt = create_prompt_from_callable("llmling.testing.processors.uppercase_text")

    assert prompt.name == "uppercase_text"
    assert "Convert text to uppercase" in prompt.description

    # Test execution
    messages = await prompt.format({"text": "test"})
    assert messages[1].get_text_content() == "TEST"


def test_create_prompt_invalid_import():
    """Test prompt creation with invalid import path."""
    with pytest.raises(ValueError, match="Could not import callable"):
        create_prompt_from_callable("nonexistent.module.function")


@pytest.mark.asyncio
async def test_argument_validation():
    """Test argument validation in created prompts."""
    prompt = create_prompt_from_callable(example_function)

    # Should fail without required argument
    with pytest.raises(ValueError, match="Missing required arguments"):
        await prompt.format({})


def test_system_message():
    """Test that system message contains function info."""
    prompt = create_prompt_from_callable(example_function)

    system_msg = prompt.messages[0]
    assert system_msg.role == "system"
    assert "Content from example_function" in system_msg.get_text_content()


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


@pytest.mark.asyncio
async def test_async_function_execution():
    """Test that async functions are properly executed."""
    prompt = create_prompt_from_callable(async_function)

    messages = await prompt.format({"content": "test", "mode": "upper"})
    assert messages[1].get_text_content() == "TEST"

    messages = await prompt.format({"content": "test"})
    assert messages[1].get_text_content() == "test"
