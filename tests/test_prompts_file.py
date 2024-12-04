"""Test file-based prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError
import pytest

from llmling.prompts.models import FilePrompt


if TYPE_CHECKING:
    import pathlib


@pytest.fixture
def temp_prompt_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a temporary prompt file."""
    prompt_file = tmp_path / "test_prompt.txt"
    prompt_file.write_text("Hello {name}!")
    return prompt_file


def test_create_file_prompt(temp_prompt_file: pathlib.Path) -> None:
    """Test creating a file-based prompt."""
    prompt = FilePrompt(
        name="test",
        description="Test prompt",
        path=temp_prompt_file,
        format="text",
        arguments=[{"name": "name", "required": True}],
    )
    assert prompt.type == "file"
    assert prompt.path == temp_prompt_file
    assert prompt.fmt == "text"


def test_invalid_file_prompt() -> None:
    """Test creating prompt with non-existent file."""
    prompt = FilePrompt(
        name="test",
        description="Test prompt",
        path="nonexistent.txt",
        format="text",
    )
    with pytest.raises(FileNotFoundError):
        assert prompt.messages


def test_invalid_format() -> None:
    """Test creating prompt with invalid format."""
    with pytest.raises(ValidationError):
        FilePrompt.model_validate({
            "name": "test",
            "description": "Test prompt",
            "path": "test.txt",
            "format": "invalid",
        })


@pytest.mark.asyncio
async def test_format_text_prompt(temp_prompt_file: pathlib.Path) -> None:
    """Test formatting a text prompt."""
    prompt = FilePrompt(
        name="test",
        description="Test prompt",
        path=temp_prompt_file,
        format="text",
        arguments=[{"name": "name", "required": True}],
    )
    messages = await prompt.format({"name": "World"})
    assert len(messages) == 1
    assert messages[0].get_text_content() == "Hello World!"


@pytest.mark.asyncio
async def test_jinja_prompt(tmp_path: pathlib.Path) -> None:
    """Test Jinja2 template prompt."""
    prompt_file = tmp_path / "test_prompt.j2"
    prompt_file.write_text("Hello {{ name }}{% if excited %}!{% endif %}")

    prompt = FilePrompt(
        name="test",
        description="Test prompt",
        path=prompt_file,
        format="jinja2",
        arguments=[
            {"name": "name", "required": True},
            {"name": "excited", "required": False},
        ],
    )

    messages = await prompt.format({"name": "World", "excited": True})
    assert messages[0].get_text_content() == "Hello World!"

    messages = await prompt.format({"name": "World"})
    assert messages[0].get_text_content() == "Hello World"


@pytest.mark.asyncio
async def test_missing_arguments(temp_prompt_file: pathlib.Path) -> None:
    """Test formatting with missing required arguments."""
    prompt = FilePrompt(
        name="test",
        description="Test prompt",
        path=temp_prompt_file,
        format="text",
        arguments=[{"name": "name", "required": True}],
    )
    with pytest.raises(ValueError, match="Missing required arguments"):
        await prompt.format({})


@pytest.mark.asyncio
async def test_undefined_argument_in_template(tmp_path: pathlib.Path) -> None:
    """Test using undefined argument in template."""
    prompt_file = tmp_path / "test_prompt.txt"
    prompt_file.write_text("Hello {name} {undefined}!")

    prompt = FilePrompt(
        name="test",
        description="Test prompt",
        path=prompt_file,
        format="text",
        arguments=[{"name": "name", "required": True}],
    )
    with pytest.raises(ValueError, match="Missing argument in template"):
        await prompt.format({"name": "World"})
