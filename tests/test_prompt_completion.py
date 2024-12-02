from __future__ import annotations

from typing import Literal, Optional

import pytest

from llmling.prompts.function import create_prompt_from_callable
from llmling.prompts.registry import PromptRegistry


def example_function(
    text: str,
    mode: Literal["simple", "detailed"] = "simple",
    tags: Literal["news", "tech", "science"] | None = None,
    active: bool = True,
    count: int = 1,
) -> str:
    """Process text with given parameters.

    Args:
        text: Input text to process
        mode: Processing mode (one of: simple, detailed)
        tags: Optional category tag
        active: Whether processing is active
        count: Number of iterations (one of: 1, 2, 3)

    Returns:
        Processed text
    """
    return text


def get_framework_completions(current: str) -> list[str]:
    """Custom completion function for frameworks."""
    frameworks = ["django", "flask", "fastapi"]
    return [f for f in frameworks if f.startswith(current)]


@pytest.fixture
def registry() -> PromptRegistry:
    """Create a prompt registry with test prompt."""
    registry = PromptRegistry()

    # Create prompt with custom completion
    prompt = create_prompt_from_callable(
        example_function,
        completions={"text": get_framework_completions},
    )
    registry["test_prompt"] = prompt

    return registry


@pytest.mark.asyncio
async def test_custom_completion(registry: PromptRegistry) -> None:
    """Test custom completion function."""
    completions = await registry.get_completions(
        current_value="dj",
        argument_name="text",
        prompt_name="test_prompt",
    )
    assert completions == ["django"]


@pytest.mark.asyncio
async def test_literal_completion(registry: PromptRegistry) -> None:
    """Test completion for Literal type."""
    completions = await registry.get_completions(
        current_value="s",
        argument_name="mode",
        prompt_name="test_prompt",
    )
    assert sorted(completions) == ["simple"]


@pytest.mark.asyncio
async def test_optional_literal_completion(registry: PromptRegistry) -> None:
    """Test completion for Optional[Literal] type."""
    completions = await registry.get_completions(
        current_value="t",
        argument_name="tags",
        prompt_name="test_prompt",
    )
    assert sorted(completions) == ["tech"]


@pytest.mark.asyncio
async def test_bool_completion(registry: PromptRegistry) -> None:
    """Test completion for bool type."""
    completions = await registry.get_completions(
        current_value="t",
        argument_name="active",
        prompt_name="test_prompt",
    )
    assert completions == ["true"]

    completions = await registry.get_completions(
        current_value="f",
        argument_name="active",
        prompt_name="test_prompt",
    )
    assert completions == ["false"]


@pytest.mark.asyncio
async def test_default_value_completion(registry: PromptRegistry) -> None:
    """Test completion includes default value."""
    completions = await registry.get_completions(
        current_value="",
        argument_name="mode",
        prompt_name="test_prompt",
    )
    assert "simple" in completions


@pytest.mark.asyncio
async def test_description_based_completion(registry: PromptRegistry) -> None:
    """Test completion from 'one of:' in description."""
    completions = await registry.get_completions(
        current_value="",
        argument_name="count",
        prompt_name="test_prompt",
    )
    assert "1" in completions  # Default
    assert all(str(x) in completions for x in (1, 2, 3))  # From description


@pytest.mark.asyncio
async def test_completion_with_no_matches(registry: PromptRegistry) -> None:
    """Test completion with no matching values."""
    completions = await registry.get_completions(
        current_value="xyz",
        argument_name="mode",
        prompt_name="test_prompt",
    )
    assert completions == []


@pytest.mark.asyncio
async def test_completion_for_unknown_argument(registry: PromptRegistry) -> None:
    """Test completion for non-existent argument."""
    completions = await registry.get_completions(
        current_value="",
        argument_name="unknown",
        prompt_name="test_prompt",
    )
    assert completions == []


@pytest.mark.asyncio
async def test_completion_for_unknown_prompt(registry: PromptRegistry) -> None:
    """Test completion for non-existent prompt."""
    completions = await registry.get_completions(
        current_value="",
        argument_name="text",
        prompt_name="unknown_prompt",
    )
    assert completions == []


def test_create_prompt_with_completions() -> None:
    """Test prompt creation with completion functions."""
    prompt = create_prompt_from_callable(
        example_function, completions={"text": get_framework_completions}
    )

    args = {arg.name: arg for arg in prompt.arguments}

    # Check custom completion function
    assert args["text"].completion_function is not None
    assert args["text"].completion_function("dj") == ["django"]

    # Check type hint preservation
    assert args["mode"].type_hint == Literal["simple", "detailed"]
    assert args["tags"].type_hint == Optional[Literal["news", "tech", "science"]]  # noqa: UP007
    assert args["active"].type_hint is bool


@pytest.mark.asyncio
async def test_combined_completions(registry: PromptRegistry) -> None:
    """Test multiple completion sources combined."""

    # Create a prompt with both custom completion and description-based options
    def process(
        text: str,
        fmt: Literal["md", "txt", "rst"] = "txt",  # Define in Literal
    ) -> str:
        """Process text.

        Args:
            text: Input text
            fmt: Output format (one of: md, txt, rst, html)
        """
        return text

    def custom_complete(current: str) -> list[str]:
        return ["html"] if current.startswith("h") else []

    prompt = create_prompt_from_callable(process, completions={"fmt": custom_complete})
    registry["combined"] = prompt

    # Should get both Literal values and custom completion
    completions = await registry.get_completions(
        current_value="",
        argument_name="fmt",
        prompt_name="combined",
    )
    assert set(completions) == {"md", "txt", "rst", "html"}


@pytest.mark.asyncio
async def test_completion_order_priority(registry: PromptRegistry) -> None:
    """Test that completion sources have correct priority."""

    def process(choice: Literal["a", "b"] = "a") -> str:
        """Process with choice.

        Args:
            choice: Choose option (one of: x, y, z)
        """
        return choice

    def custom_complete(current: str) -> list[str]:
        return ["custom"]

    prompt = create_prompt_from_callable(process, completions={"choice": custom_complete})
    registry["priority"] = prompt

    completions = await registry.get_completions(
        current_value="",
        argument_name="choice",
        prompt_name="priority",
    )
    completions_list = list(completions)

    assert completions_list[0] == "custom"  # Custom completion first
    assert all(x in completions_list for x in ["a", "b"])  # Literal values included
    assert all(
        x in completions_list for x in ["x", "y", "z"]
    )  # Description values included
