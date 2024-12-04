"""Prompt models for LLMling."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Annotated, Any, Literal, get_type_hints

from docstring_parser import parse as parse_docstring
from pydantic import BaseModel, ConfigDict, Field

from llmling.completions import CompletionFunction  # noqa: TC001
from llmling.core.typedefs import MessageContent, MessageRole
from llmling.utils import calling, importing


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class ExtendedPromptArgument(BaseModel):
    """Prompt argument with validation information."""

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


class BasePrompt(BaseModel):
    """Base class for all prompts."""

    name: str
    description: str
    arguments: list[ExtendedPromptArgument] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # messages: list[PromptMessage]

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
        """Format this prompt with given arguments.

        Args:
            arguments: Optional argument values

        Returns:
            List of formatted messages

        Raises:
            ValueError: If required arguments are missing
        """
        raise NotImplementedError


class StaticPrompt(BasePrompt):
    """Static prompt defined by message list."""

    messages: list[PromptMessage]
    type: Literal["static"] = Field("static", init=False)

    async def format(
        self, arguments: dict[str, Any] | None = None
    ) -> list[PromptMessage]:
        """Format static prompt messages with arguments."""
        args = arguments or {}
        self.validate_arguments(args)

        # Add default values for optional arguments
        for arg in self.arguments:
            if arg.name not in args and not arg.required:
                args[arg.name] = arg.default if arg.default is not None else ""

        # Format all messages
        formatted_messages = []
        for msg in self.messages:
            match msg.content:
                case str():
                    content: MessageContent | list[MessageContent] = MessageContent(
                        type="text", content=msg.content.format(**args)
                    )
                case MessageContent() if msg.content.type == "text":
                    content = MessageContent(
                        type="text", content=msg.content.content.format(**args)
                    )
                case list():
                    content = [
                        MessageContent(
                            type=item.type,
                            content=item.content.format(**args)
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


class DynamicPrompt(BasePrompt):
    """Dynamic prompt loaded from callable."""

    import_path: str
    template: str | None = None
    completions: dict[str, str] | None = None
    type: Literal["dynamic"] = Field("dynamic", init=False)

    @property
    def messages(self) -> list[PromptMessage]:
        """Get the template messages for this prompt.

        Note: These are template messages - actual content will be populated
        during format() when the callable is executed.
        """
        template = self.template or "{result}"
        return [
            PromptMessage(
                role="system",
                content=MessageContent(type="text", content=f"Content from {self.name}:"),
            ),
            PromptMessage(
                role="user",
                content=MessageContent(type="text", content=template),
            ),
        ]

    async def format(
        self, arguments: dict[str, Any] | None = None
    ) -> list[PromptMessage]:
        args = arguments or {}
        self.validate_arguments(args)

        try:
            result = await calling.execute_callable(self.import_path, **args)
            # Use result directly in template
            template = self.template or "{result}"
            content = MessageContent(
                type="text",
                content=template.format(result=result),  # Format with result
            )

            return [
                PromptMessage(
                    role="system",
                    content=MessageContent(
                        type="text", content=f"Content from {self.name}:"
                    ),
                ),
                PromptMessage(
                    role="user",
                    content=content,  # Use formatted content
                ),
            ]
        except Exception as exc:
            msg = f"Failed to execute prompt callable: {exc}"
            raise ValueError(msg) from exc

    @classmethod
    def from_callable(
        cls,
        fn: Callable[..., Any] | str,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        template_override: str | None = None,
        completions: Mapping[str, CompletionFunction] | None = None,
    ) -> DynamicPrompt:
        """Create a prompt from a callable.

        Args:
            fn: Function or import path to create prompt from
            name_override: Optional override for prompt name
            description_override: Optional override for prompt description
            template_override: Optional override for message template
            completions: Optional dict mapping argument names to completion functions

        Returns:
            DynamicPrompt instance

        Raises:
            ValueError: If callable cannot be imported or is invalid
        """
        completions = completions or {}
        # Import if string path provided
        if isinstance(fn, str):
            fn = importing.import_callable(fn)

        # Get function metadata
        name = name_override or fn.__name__
        sig = inspect.signature(fn)
        hints = get_type_hints(fn, include_extras=True)

        # Parse docstring
        docstring = inspect.getdoc(fn)
        if docstring:
            parsed = parse_docstring(docstring)
            description = description_override or parsed.short_description
            # Create mapping of param names to descriptions
            arg_docs = {
                param.arg_name: param.description
                for param in parsed.params
                if param.arg_name and param.description
            }
        else:
            description = description_override or f"Prompt from {name}"
            arg_docs = {}

        # Create arguments
        arguments = []
        for param_name, param in sig.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            type_hint = hints.get(param_name, Any)
            required = param.default == param.empty
            arg = ExtendedPromptArgument(
                name=param_name,
                description=arg_docs.get(param_name),
                required=required,
                type_hint=type_hint,
                default=None if param.default is param.empty else param.default,
                completion_function=completions.get(param_name),
            )
            arguments.append(arg)

        path = f"{fn.__module__}.{fn.__qualname__}"
        return cls(
            name=name,
            description=description or "",
            arguments=arguments,
            import_path=path,
            template=template_override,
            metadata={"source": "function", "import_path": path},
        )


# Type to use in configuration
PromptType = Annotated[StaticPrompt | DynamicPrompt, Field(discriminator="type")]
