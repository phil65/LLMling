"""Prompt models for MCP."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llmling.completions import CompletionFunction  # noqa: TC001
from llmling.core.typedefs import MessageContent, MessageRole


class ExtendedPromptArgument(BaseModel):
    """Prompt argument with validation information.

    Extends MCP's prompt argument with additional metadata for validation.
    """

    name: str
    description: str | None = None
    required: bool = False
    type_hint: Any = str
    default: Any | None = None
    completion_function: CompletionFunction = None

    model_config = ConfigDict(frozen=True)


class PromptMessage(BaseModel):
    """A message in a prompt template."""

    role: MessageRole
    content: str | MessageContent | list[MessageContent] = ""

    model_config = ConfigDict(frozen=True)

    def get_text_content(self) -> str:
        """Get text content of message."""
        match self.content:
            case str():
                return self.content
            case MessageContent() if self.content.type == "text":
                return self.content.content
            case list() if self.content:
                # Join text content items with space
                text_items = [
                    item.content
                    for item in self.content
                    if isinstance(item, MessageContent) and item.type == "text"
                ]
                return " ".join(text_items) if text_items else ""
            case _:
                return ""


class Prompt(BaseModel):
    """MCP prompt template."""

    name: str
    description: str
    messages: list[PromptMessage]
    arguments: list[ExtendedPromptArgument] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    def validate_arguments(self, provided: dict[str, Any]) -> None:
        """Validate that required arguments are provided."""
        required = {arg.name for arg in self.arguments if arg.required}
        missing = required - set(provided)
        if missing:
            msg = f"Missing required arguments: {', '.join(missing)}"
            raise ValueError(msg)

    async def format(
        self, arguments: dict[str, Any] | None = None
    ) -> list[PromptMessage]:
        """Format prompt messages with arguments.

        Args:
            arguments: Values to format the prompt with

        Returns:
            List of formatted messages

        Raises:
            ValueError: If required arguments are missing
        """
        args = arguments or {}
        self.validate_arguments(args)

        # If this is a function prompt, execute it
        if self.metadata.get("source") == "function":
            try:
                from llmling.utils import calling

                import_path = self.metadata["import_path"]
                result = await calling.execute_callable(import_path, **args)
                format_args = {"result": result}
            except Exception as exc:  # noqa: BLE001
                format_args = {"result": f"Error executing function: {exc}"}
        else:
            # Add default values for optional arguments
            format_args = args.copy()  # Make a copy to avoid modifying input
            for arg in self.arguments:
                if arg.name not in format_args:
                    if arg.default is not None:
                        format_args[arg.name] = arg.default
                    elif not arg.required:
                        format_args[arg.name] = ""  # Empty string for optional args

        # Format all messages
        formatted_messages = []
        for msg in self.messages:
            match msg.content:
                case str():
                    content: MessageContent | list[MessageContent] = MessageContent(
                        type="text", content=msg.content.format(**format_args)
                    )
                case MessageContent() if msg.content.type == "text":
                    content = MessageContent(
                        type="text", content=msg.content.content.format(**format_args)
                    )
                case list():
                    content = [
                        MessageContent(
                            type=item.type,
                            content=item.content.format(**format_args)
                            if item.type == "text"
                            else item.content,
                            alt_text=item.alt_text,
                        )
                        for item in msg.content
                        if isinstance(item, MessageContent)
                    ]
                case _:
                    content = msg.content

            formatted_messages.append(PromptMessage(role=msg.role, content=content))

        return formatted_messages
