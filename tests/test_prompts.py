from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
import pytest

from llmling.config.models import Config
from llmling.core.exceptions import ProcessorError
from llmling.prompts.models import (
    ArgumentType,
    ExtendedPromptArgument,
    MessageContent,
    Prompt,
    PromptMessage,
    PromptResult,
)
from llmling.prompts.rendering import render_prompt
from llmling.resources.base import ResourceLoader
from llmling.resources.models import LoadedResource, ResourceMetadata


if TYPE_CHECKING:
    from llmling.processors.registry import ProcessorRegistry


def test_config_with_prompts():
    """Test config with prompts section."""
    config_data = {
        "version": "1.0",
        # Need to add required fields from Config model
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
        "resources": {},  # Required field
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
    assert "review" in config.prompts
    msg = config.prompts["review"].messages[0]
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
        type=ArgumentType.TEXT,
        description="Name to greet",
        required=True,
    )
    return Prompt(
        name="test",
        description="Test prompt",
        messages=[PromptMessage(role="user", content="Hello {name}")],
        arguments=[arg],
    )


@pytest.fixture
def resource_prompt() -> Prompt:
    """Create a prompt with resource references."""
    return Prompt(
        name="analyze",
        description="Analyze code",
        messages=[
            PromptMessage(
                role="user",
                content=[
                    MessageContent.text("Analyze this code:"),
                    MessageContent.resource("source://test.py"),
                ],
            )
        ],
    )


@pytest.fixture
def loaded_resource() -> LoadedResource:
    """Create a sample loaded resource."""
    return LoadedResource(
        content="def test(): pass",
        source_type="source",
        metadata=ResourceMetadata(
            uri="source://test.py", mime_type="text/x-python", name="test.py"
        ),
    )


async def test_prompt_rendering(sample_prompt: Prompt):
    """Test basic prompt rendering."""
    result = await render_prompt(sample_prompt, {"name": "World"})
    assert isinstance(result, PromptResult)
    assert len(result.messages) == 1
    assert isinstance(result.messages[0].content, list)
    assert result.messages[0].content[0].content == "Hello World"


async def test_prompt_with_resources(
    resource_prompt: Prompt,
    loaded_resource: LoadedResource,
):
    """Test prompt rendering with resource resolution."""

    class MockProcessorRegistry:
        async def process(self, content: str, *args: Any, **kwargs: Any) -> str:
            return "processed content"

    class MockLoader(ResourceLoader[None]):
        context_class = type(None)
        uri_scheme = "source"

        def __init__(self) -> None:
            super().__init__(None)

        async def _load_impl(
            self,
            resource: Any,
            name: str,
            processor_registry: ProcessorRegistry | None,
        ) -> LoadedResource:
            return loaded_resource

    class MockResourceRegistry:
        # Changed to sync method to match the actual interface
        def find_loader_for_uri(self, uri: str) -> ResourceLoader[Any]:
            return MockLoader()

    result = await render_prompt(
        resource_prompt,
        {},
        resource_registry=MockResourceRegistry(),
        processor_registry=MockProcessorRegistry(),
    )

    assert len(result.messages) == 1
    msg = result.messages[0]
    assert isinstance(msg.content, list)
    assert len(msg.content) == 2  # noqa: PLR2004
    assert msg.content[0].type == "text"
    assert msg.content[0].content == "Analyze this code:"
    assert msg.content[1].type == "resource"
    assert msg.resolved_content is not None
    assert len(msg.resolved_content) == 2  # noqa: PLR2004
    assert msg.resolved_content[0].original == msg.content[0]
    assert msg.resolved_content[1].resolved == loaded_resource


async def test_prompt_validation():
    """Test prompt argument validation."""
    arg = ExtendedPromptArgument(
        name="required_arg",
        type=ArgumentType.TEXT,
        required=True,
    )
    prompt = Prompt(
        name="test",
        description="",
        messages=[PromptMessage(role="user", content="Test {required_arg}")],
        arguments=[arg],
    )

    with pytest.raises(ProcessorError, match="Missing required argument"):
        await render_prompt(prompt, {})
