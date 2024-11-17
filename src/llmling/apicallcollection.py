"""Module for managing API call collections."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from jinjarope.llm.calls import APICall
from jinjarope.llm.calls.completion import CompletionCall, CompletionParameters
from jinjarope.llm.prompts import ImageContent, PromptType
from jinjarope.llm.sources import (
    ContextSource,
)
from jinjarope.llm.tools import ToolDefinition, registry


if TYPE_CHECKING:
    from jinjarope.llm.tools import ToolDefinition


class LLMParameters(BaseModel):
    model: str
    """Name/ID of the LLM model to use"""

    temperature: float = Field(default=0.7, ge=0, le=1)
    """Sampling temperature (0-1) controlling randomness"""

    max_tokens: int | None = None
    """Maximum tokens in response"""

    top_p: float | None = None
    """Nucleus sampling parameter"""

    presence_penalty: float | None = None
    """Penalty for new token presence"""

    frequency_penalty: float | None = None
    """Penalty for token frequency"""

    response_format: type[BaseModel] | str | None = Field(default=None)
    """Format for model responses. Can be a Pydantic model class for structured output,
    'json_object' for JSON responses, or None for plain text."""

    json_schema: dict[str, Any] | None = None
    """Optional JSON schema for structured outputs. Only used when response_format is 'json_object'."""



class ContextBundle(BaseModel):
    bundle_id: UUID = Field(default_factory=uuid4)
    name: str
    description: str | None = None
    sources: list[ContextSource]


class APICallCollection(BaseModel):
    name: str
    description: str | None = None
    calls: list[APICall]
    context_bundles: list[ContextBundle] = Field(default_factory=list)
    tool_definitions: list[ToolDefinition] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        # Register tools from definitions
        for tool in self.tool_definitions:
            registry.register(tool)


if __name__ == "__main__":
    # Create a simple test case for CompletionCall
    test_completion_call = CompletionCall(
        name="Test Vision",
        description="A test of the vision system",
        prompts=[
            # TextContent(
            #     type=PromptType.SYSTEM, text="You are a weather assistant.", order=0
            # ),
            ImageContent(
                type=PromptType.USER, url="https://www.dhs.wisconsin.gov/sites/default/files/styles/large/public/dam/image/5/thermometer-inthe-snow.jpg?itok=rWkiCLO_", order=1
            ),
            # TextContent(
            #     type=PromptType.USER, text="What can you see?.", order=0
            # ),
        ],
        parameters=CompletionParameters(
            model="dall-e-3",
            temperature=0.7,
        ),
    )
    print(test_completion_call.execute())

    # # Create a simple test case for ImageGenerationCall
    # test_image_call = ImageGenerationCall(
    #     name="Test Image Processing",
    #     description="A basic test of the image API call system",
    #     image_url="https://picsum.photos/100/100",
    #     parameters=ImageParameters(
    #         resolution="1080p",
    #         format="JPEG",
    #         quality="standard",
    #         color_mode="RGB",
    #     ),
    #     context_sources=[
    #         ApiSource(
    #             source_type=SourceType.API,
    #             name="Image Metadata",
    #             url="https://picsum.photos/100/100",
    #         ),
    #     ],
    # )
    # print(test_image_call.execute())
