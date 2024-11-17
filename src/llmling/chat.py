"""LiteLLM wrapper providing an elegant, object-oriented interface for AI completions."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
import hashlib
import inspect
import json
import os
from typing import Any, Literal, get_type_hints

import diskcache
import docstring_parser
import litellm
import pydantic
from pydantic import BaseModel
from typing_extensions import ParamSpec, TypedDict
from upath import UPath
import yamling

from llmling import jsonschema


type MessageRole = Literal["system", "user", "assistant", "function"]
P = ParamSpec("P")


class ContentType(BaseModel):
    """Base class for different types of content."""

    type: str


class TextContent(ContentType):
    """Text content for messages."""

    type: Literal["text"] = "text"
    text: str

    @classmethod
    def from_str(cls, text: str) -> TextContent:
        """Creates a TextContent from a string."""
        return cls(text=text)


class ImageContent(ContentType):
    """Image content for vision models."""

    type: Literal["image"] = "image"
    url: str
    detail: Literal["low", "high", "auto"] = "auto"


class ToolCallArguments(TypedDict):
    """Arguments for a tool call."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Represents a tool call from the model."""

    id: str
    type: Literal["function"] = "function"
    function: ToolCallArguments


class Tool(BaseModel):
    """Represents a function tool that can be used by the model."""

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any] | None = None

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_function(cls, func: Callable[P, Any]) -> Tool:
        """Creates a Tool from a Python function."""
        signature = inspect.signature(func)
        docstring = docstring_parser.parse(func.__doc__ or "")
        type_hints = get_type_hints(func)

        parameters = {"type": "object", "properties": {}, "required": []}

        for param_name, param in signature.parameters.items():
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)

            param_type = type_hints.get(param_name, Any)
            param_doc = next(
                (p.description for p in docstring.params if p.arg_name == param_name),
                None,
            )

            param_info = {
                "type": jsonschema.python_type_to_json_schema(param_type),
                "description": param_doc or f"Parameter {param_name}",
            }

            parameters["properties"][param_name] = param_info

        return cls(
            name=func.__name__,
            description=docstring.short_description or func.__name__,
            parameters=parameters,
            function=func,
        )


class Message(BaseModel):
    """A chat message that can contain multiple types of content."""

    role: MessageRole
    content: list[ContentType] | str
    name: str | None = None
    tool_calls: list[ToolCall] | None = None

    @classmethod
    def system(cls, text: str) -> Message:
        """Creates a system message."""
        return cls(role="system", content=text)

    @classmethod
    def user(cls, content: str | list[ContentType]) -> Message:
        """Creates a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, text: str) -> Message:
        """Creates an assistant message."""
        return cls(role="assistant", content=text)

    @classmethod
    def tool_result(cls, name: str, result: str) -> Message:
        """Creates a tool result message."""
        return cls(role="function", content=result, name=name)


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Metadata(BaseModel):
    """Metadata about the completion."""

    model: str
    created_at: datetime
    usage: Usage
    cache_hit: bool = False


class CompletionResult(BaseModel):
    """A structured completion result."""

    content: str | None
    tool_calls: list[tuple[Callable[..., Any], dict[str, Any]]] | None = None
    metadata: Metadata
    raw_response: dict[str, Any]


class Chat:
    """Main chat interface for interacting with language models."""

    def __init__(
        self,
        model: str,
        cache_dir: str | UPath | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = diskcache.Cache(str(cache_dir)) if cache_dir else None
        self._tool_registry: dict[str, Tool] = {}
        self._conversation: list[Message] = []

    def add_tool(self, tool: Tool | Callable[..., Any]) -> Chat:
        """Adds a tool to the chat."""
        if callable(tool):
            tool = Tool.from_function(tool)
        self._tool_registry[tool.name] = tool
        return self

    def add_message(self, message: Message) -> Chat:
        """Adds a message to the conversation."""
        self._conversation.append(message)
        return self

    def _create_completion_params(self) -> dict[str, Any]:
        """Creates the parameters for the completion request."""
        params = {
            "model": self.model,
            "messages": [msg.model_dump() for msg in self._conversation],
            "temperature": self.temperature,
        }

        if self.max_tokens:
            params["max_tokens"] = self.max_tokens

        if self._tool_registry:
            params["tools"] = [
                tool.model_dump(exclude={"function"})
                for tool in self._tool_registry.values()
            ]

        return params

    def _process_tool_calls(
        self, response: dict
    ) -> list[tuple[Callable[..., Any], dict[str, Any]]] | None:
        """Processes tool calls from the response."""
        if not (
            tool_calls := response.get("choices", [{}])[0]
            .get("message", {})
            .get("tool_calls")
        ):
            return None

        result = []
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            if tool := self._tool_registry.get(tool_name):
                arguments = json.loads(tool_call["function"]["arguments"])
                result.append((tool.function, arguments))

        return result if result else None

    def _create_metadata(self, response: dict, cache_hit: bool = False) -> Metadata:
        """Creates metadata from the response."""
        return Metadata(
            model=response.get("model", self.model),
            created_at=datetime.fromtimestamp(
                response.get("created", datetime.now().timestamp())
            ),
            usage=Usage(
                **response.get(
                    "usage",
                    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                )
            ),
            cache_hit=cache_hit,
        )

    def complete(self, cache_ttl: timedelta | None = None) -> CompletionResult:
        """Performs the completion request.

        Args:
            cache_ttl: Optional cache duration

        Returns:
            A structured completion result
        """
        params = self._create_completion_params()

        if cache_ttl and self.cache:
            cache_key = hashlib.sha256(
                json.dumps(params, sort_keys=True).encode()
            ).hexdigest()

            response = self.cache.get(
                cache_key, default=None, expire=int(cache_ttl.total_seconds())
            )

            if response:
                return CompletionResult(
                    content=response["choices"][0]["message"].get("content"),
                    tool_calls=self._process_tool_calls(response),
                    metadata=self._create_metadata(response, cache_hit=True),
                    raw_response=response,
                )

        response = litellm.completion(**params)

        if cache_ttl and self.cache:
            self.cache.set(cache_key, response, expire=int(cache_ttl.total_seconds()))

        return CompletionResult(
            content=response["choices"][0]["message"].get("content"),
            tool_calls=self._process_tool_calls(response),
            metadata=self._create_metadata(response),
            raw_response=response,
        )

    @staticmethod
    def from_yaml(file_path: str) -> Chat:
        """Creates a Chat instance from a YAML configuration file."""
        text = UPath(file_path).read_text()
        config = yamling.load_yaml(text)

        chat = ChatBuilder(config["model"]).build()

        for tool_def in config.get("tool_definitions", []):
            tool = Tool(
                name=tool_def["name"],
                description=tool_def["description"],
                parameters=tool_def["parameters"],
                function=None,  # Assuming function is defined elsewhere
            )
            chat.add_tool(tool)

        for call in config.get("calls", []):
            for prompt in call["prompts"]:
                message = Message(role=prompt["type"], content=prompt["text"])
                chat.add_message(message)

        return chat


class ChatBuilder:
    """Fluent builder for creating chat instances."""

    def __init__(self, model: str):
        self._model = model
        self._temperature = 1.0
        self._max_tokens: int | None = None
        self._cache_dir: UPath | None = None

    def temperature(self, value: float) -> ChatBuilder:
        """Sets the temperature."""
        self._temperature = value
        return self

    def max_tokens(self, value: int) -> ChatBuilder:
        """Sets the max tokens."""
        self._max_tokens = value
        return self

    def with_cache(self, cache_dir: os.PathLike[str] | UPath) -> ChatBuilder:
        """Enables caching."""
        self._cache_dir = UPath(cache_dir)
        return self

    def build(self) -> Chat:
        """Creates the Chat instance."""
        return Chat(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            cache_dir=self._cache_dir,
        )


# Usage example
if __name__ == "__main__":
    chat = Chat.from_yaml("/path/to/prompts/example_collection.yml")
    result = chat.complete()
    print(f"Model response: {result.content}")

    # from datetime import timedelta

    # def calculate_area(radius: float) -> float:
    #     """Calculates circle area.

    #     Args:
    #         radius: Circle radius in meters
    #     """
    #     import math
    #     return math.pi * radius ** 2

    # # Create a chat instance using the builder
    # chat = (ChatBuilder("gpt-4-vision-preview")
    #         .temperature(0.7)
    #         .with_cache("./cache")
    #         .build())

    # # Add a tool
    # chat.add_tool(calculate_area)

    # # Create a conversation with different types of content
    # chat.add_message(Message.system("You are a helpful assistant."))

    # # Add a message with both image and text
    # chat.add_message(Message.user([
    #     ImageContent(url="https://example.com/circle.jpg", detail="high"),
    #     TextContent(text="What's the area of this circle? The radius is 5 meters.")
    # ]))

    # # Get completion with caching
    # result = chat.complete(cache_ttl=timedelta(hours=1))

    # # Process the result
    # print(f"Model response: {result.content}")
    # print(f"Model: {result.metadata.model}")
    # print(f"Token usage: {result.metadata.usage.total_tokens}")
    # print(f"Cache hit: {result.metadata.cache_hit}")

    # # Handle tool calls
    # if result.tool_calls:
    #     for func, args in result.tool_calls:
    #         output = func(**args)
    #         # Add the tool result back to the conversation
    #         chat.add_message(Message.tool_result(func.__name__, str(output)))

    #     # Get final response
    #     final_result = chat.complete(cache_ttl=timedelta(hours=1))
    #     print(f"Final response: {final_result.content}")
