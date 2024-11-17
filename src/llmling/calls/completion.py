"""Completion API call implementation."""

import json
from typing import Any, Literal

import litellm
from litellm.utils import CustomStreamWrapper
from pydantic import BaseModel, Field

from jinjarope.llm.calls.base import APICall
from jinjarope.llm.prompts import BasePrompt, PromptType, TextContent
from jinjarope.llm.tools import ToolDefinition, ToolResponse, registry


class CompletionParameters(BaseModel):
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

    tools: list[str] | None = None
    """Names of tools to make available"""

    tool_choice: str | None = None
    """How to handle tool selection (none/auto/specific)"""

    response_format: type[BaseModel] | dict[str, Any] | Literal["json_object"] | None = (
        Field(default=None)
    )
    """Format for model responses. Can be:
    - A Pydantic model class for structured output
    - A dict containing a JSON schema
    - 'json_object' for generic JSON responses
    - None for plain text"""


class CompletionCall(APICall):
    prompts: list[BasePrompt]
    """Ordered list of prompts to send"""

    parameters: CompletionParameters
    """Completion-specific parameters for this call"""

    def execute(self) -> dict:
        """Execute the completion API call."""
        messages = [
            prompt.to_message_content()
            for prompt in sorted(self.prompts, key=lambda p: p.order)
        ]

        # Handle response format
        if isinstance(self.parameters.response_format, type) and issubclass(
            self.parameters.response_format, BaseModel
        ):
            # Using Pydantic model
            model_class = self.parameters.response_format
            schema = model_class.model_json_schema()
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "You must respond with JSON matching this Pydantic model schema:\n"
                        f"{json.dumps(schema, indent=2)}\n"
                        "The response must be valid JSON that can be parsed into this model."
                    ),
                },
            )
            params_response_format = {"type": "json_object"}
        elif isinstance(self.parameters.response_format, dict):
            # Using provided JSON schema
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "You must respond with JSON matching this schema:\n"
                        f"{json.dumps(self.parameters.response_format, indent=2)}\n"
                        "The response must be valid JSON that matches this schema."
                    ),
                },
            )
            params_response_format = {"type": "json_object"}
        elif self.parameters.response_format == "json_object":
            # Using JSON object format
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "You must respond with a valid JSON object.",
                },
            )
            params_response_format = {"type": "json_object"}
        else:
            # Text format
            params_response_format = None

        # Add tool definitions if specified
        tools = None
        tool_choice = None
        if self.parameters.tools:
            tools = registry.get_definitions()
            tool_choice = self.parameters.tool_choice or "auto"

        # Prepare parameters
        params = self.parameters.model_dump(
            exclude={"tools", "tool_choice", "response_format"}
        )
        if params_response_format:
            params["response_format"] = params_response_format

        try:
            response = litellm.completion(
                messages=messages, tools=tools, tool_choice=tool_choice, **params
            )

            result = {
                "call_id": str(self.call_id),
                "name": self.name,
                "description": self.description,
                "response": response,
                "parameters": self.parameters.model_dump(),
                "context_sources": [
                    source.model_dump() for source in self.context_sources
                ],
            }

            # Parse response if using Pydantic model
            if isinstance(self.parameters.response_format, type) and issubclass(
                self.parameters.response_format, BaseModel
            ):
                try:
                    content = None
                    if isinstance(response, CustomStreamWrapper):
                        # Access streaming response content
                        content = (
                            response.choices[0]["message"]["content"]
                            if response.choices
                            else None
                        )
                    else:
                        # Access regular response content
                        content = (
                            response.choices[0].message.content
                            if hasattr(response, "choices")
                            else None
                        )

                    if content:
                        parsed_response = (
                            self.parameters.response_format.model_validate_json(content)
                        )
                        result["parsed_response"] = parsed_response.model_dump()
                except Exception as e:
                    result["parsing_error"] = str(e)
            else:
                return result

        except Exception as e:
            return {"call_id": str(self.call_id), "error": str(e), "success": False}

    def handle_tool_call(self, name: str, arguments: dict) -> ToolResponse:
        """Execute a tool call.

        Args:
            name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            The tool's response

        Raises:
            ValueError: If the tool is not found or not enabled
        """
        if not self.parameters.tools or name not in self.parameters.tools:
            msg = f"Tool '{name}' not enabled for this call"
            raise ValueError(msg)

        tool = registry.get_tool(name)
        if not tool:
            msg = f"Tool '{name}' not found"
            raise ValueError(msg)

        if isinstance(tool, ToolDefinition):
            if not tool.implementation:
                msg = f"No implementation defined for tool '{name}'"
                raise ValueError(msg)

            # Import and instantiate the implementation class
            module_path, class_name = tool.implementation.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            tool_cls = getattr(module, class_name)
            tool_instance = tool_cls()
        else:
            tool_instance = tool()

        return tool_instance.execute(**arguments)


if __name__ == "__main__":
    import json

    from pydantic import BaseModel, Field

    # Example Pydantic model for structured output
    class WeatherResponse(BaseModel):
        location: str
        temperature: float = Field(description="Temperature in Celsius")
        conditions: str
        forecast: list[dict[str, Any]]

    # Example 1: Basic completion call with tool
    call = CompletionCall(
        name="Weather Query",
        description="Ask about weather",
        prompts=[
            TextContent(
                type=PromptType.SYSTEM, text="You are a weather assistant.", order=0
            ),
            TextContent(
                type=PromptType.USER, text="What's the weather in London?", order=1
            ),
        ],
        parameters=CompletionParameters(
            model="gemini/gemini-1.5-flash",
            temperature=0.7,
            tools=["get_current_weather"],
            tool_choice="auto",
        ),
    )

    # Example 2: JSON schema completion call
    weather_schema = {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "temperature": {"type": "number"},
            "conditions": {"type": "string"},
            "forecast": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"day": {"type": "string"}, "temp": {"type": "number"}},
                },
            },
        },
    }

    json_call = CompletionCall(
        name="JSON Weather",
        description="Get weather info in JSON format",
        prompts=[
            TextContent(
                type=PromptType.SYSTEM,
                text="You provide weather information in structured format.",
                order=0,
            ),
            TextContent(
                type=PromptType.USER, text="What's the weather in London?", order=1
            ),
        ],
        parameters=CompletionParameters(
            model="openai/gpt-3.5-turbo", temperature=0.7, response_format=weather_schema
        ),
    )

    # Example 3: Pydantic model completion call
    pydantic_call = CompletionCall(
        name="Pydantic Weather",
        description="Get weather info using Pydantic model",
        prompts=[
            TextContent(
                type=PromptType.SYSTEM,
                text="You provide weather information in structured format.",
                order=0,
            ),
            TextContent(
                type=PromptType.USER, text="What's the weather in London?", order=1
            ),
        ],
        parameters=CompletionParameters(
            model="openai/gpt-3.5-turbo", temperature=0.7, response_format=WeatherResponse
        ),
    )

    # Execute calls
    print("Basic response:", call.execute())
    print("\nJSON response:", json_call.execute())
    print("\nPydantic response:", pydantic_call.execute())
