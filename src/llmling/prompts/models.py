"""Prompt models for LLMling."""

from __future__ import annotations

import inspect
import os  # noqa: TC003
from typing import TYPE_CHECKING, Annotated, Any, Literal, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, ImportString

from llmling.completions import CompletionFunction  # noqa: TC001
from llmling.core.log import get_logger
from llmling.core.typedefs import MessageContent, MessageRole
from llmling.utils import calling, importing


logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class PromptParameter(BaseModel):
    """Prompt argument with validation information."""

    name: str
    """Name of the argument as used in the prompt."""

    description: str | None = None
    """Human-readable description of the argument."""

    required: bool = False
    """Whether this argument must be provided when formatting the prompt."""

    type_hint: ImportString = Field(default="str")
    """Type annotation for the argument, defaults to str."""

    default: Any | None = None
    """Default value if argument is optional."""

    completion_function: ImportString | None = Field(default=None)
    """Optional function to provide argument completions."""

    model_config = ConfigDict(use_attribute_docstrings=True)


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

    name: str | None = Field(None, exclude=True)
    """Technical identifier (automatically set from config key during registration)."""

    description: str
    """Human-readable description of what this prompt does."""

    arguments: list[PromptParameter] = Field(default_factory=list)
    """List of arguments that this prompt accepts."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Additional metadata for storing custom prompt information."""
    # messages: list[PromptMessage]

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    def get_messages(self) -> list[PromptMessage]:
        """Get the messages for this prompt."""
        if not hasattr(self, "messages"):
            msg = f"{self.__class__.__name__} must implement 'messages' attribute"
            raise NotImplementedError(msg)
        return self.messages  # pyright: ignore

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
    """List of messages that make up this prompt."""

    type: Literal["text"] = Field("text", init=False)
    """Discriminator field identifying this as a static text prompt."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

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
                    msg_content = msg.content.content.format(**args)
                    content = MessageContent(type="text", content=msg_content)
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
    """Dotted import path to the callable that generates the prompt."""

    template: str | None = None
    """Optional template string for formatting the callable's output."""

    completions: dict[str, str] | None = None
    """Optional mapping of argument names to completion functions."""

    type: Literal["function"] = Field("function", init=False)
    """Discriminator field identifying this as a function-based prompt."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    @property
    def messages(self) -> list[PromptMessage]:
        """Get the template messages for this prompt.

        Note: These are template messages - actual content will be populated
        during format() when the callable is executed.
        """
        template = self.template or "{result}"
        sys_content = MessageContent(type="text", content=f"Content from {self.name}:")
        user_content = MessageContent(type="text", content=template)
        return [
            PromptMessage(role="system", content=sys_content),
            PromptMessage(role="user", content=user_content),
        ]

    def get_completion_functions(self) -> dict[str, CompletionFunction]:
        """Resolve completion function import paths and return a completion fn dict."""
        completion_funcs: dict[str, CompletionFunction] = {}
        if not self.completions:
            return {}
        for arg_name, import_path in self.completions.items():
            try:
                func = importing.import_callable(import_path)
                completion_funcs[arg_name] = func
            except ValueError:
                msg = "Failed to import completion function for %s: %s"
                logger.warning(msg, arg_name, import_path)
        return completion_funcs

    async def format(
        self, arguments: dict[str, Any] | None = None
    ) -> list[PromptMessage]:
        """Format this prompt with given arguments."""
        args = arguments or {}
        self.validate_arguments(args)

        try:
            result = await calling.execute_callable(self.import_path, **args)
            # Use result directly in template
            template = self.template or "{result}"
            msg = template.format(result=result)
            content = MessageContent(type="text", content=msg)
            msg = f"Content from {self.name}:"
            sys_content = MessageContent(type="text", content=msg)
            return [
                PromptMessage(role="system", content=sys_content),
                PromptMessage(role="user", content=content),
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
        from docstring_parser import parse as parse_docstring

        completions = completions or {}
        # Import if string path provided
        if isinstance(fn, str):
            fn = importing.import_callable(fn)

        # Get function metadata
        name = name_override or fn.__name__
        sig = inspect.signature(fn)
        hints = get_type_hints(fn, include_extras=True, localns=locals())

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
            arg = PromptParameter(
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


class FilePrompt(BasePrompt):
    """Prompt loaded from a file.

    This type of prompt loads its content from a file, allowing for longer or more
    complex prompts to be managed in separate files. The file content is loaded
    and parsed according to the specified format.
    """

    path: str | os.PathLike[str]
    """Path to the file containing the prompt content."""

    fmt: Literal["text", "markdown", "jinja2"] = Field("text", alias="format")
    """Format of the file content (text, markdown, or jinja2 template)."""

    type: Literal["file"] = Field("file", init=False)
    """Discriminator field identifying this as a file-based prompt."""

    watch: bool = False
    """Whether to watch the file for changes and reload automatically."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    @property
    def messages(self) -> list[PromptMessage]:
        """Get messages from file content."""
        import upath

        content = upath.UPath(self.path).read_text("utf-8")

        match self.fmt:
            case "text":
                # Simple text format - whole file as user message
                msg = MessageContent(type="text", content=content)
            case "markdown":
                # TODO: Parse markdown sections into separate messages
                msg = MessageContent(type="text", content=content)
            case "jinja2":
                # Raw template - will be formatted during format()
                msg = MessageContent(type="text", content=content)
            case _:
                msg = f"Unsupported format: {self.fmt}"
                raise ValueError(msg)
        return [PromptMessage(role="user", content=msg)]

    async def format(
        self, arguments: dict[str, Any] | None = None
    ) -> list[PromptMessage]:
        """Format the file content with arguments."""
        import upath

        args = arguments or {}
        self.validate_arguments(args)

        # Add default values for optional arguments
        for arg in self.arguments:
            if arg.name not in args and not arg.required:
                args[arg.name] = arg.default if arg.default is not None else ""

        content = upath.UPath(self.path).read_text("utf-8")

        if self.fmt == "jinja2":
            # Use jinja2 for template formatting
            import jinja2

            env = jinja2.Environment(autoescape=True, enable_async=True)
            template = env.from_string(content)
            content = await template.render_async(**args)
        else:
            # Use simple string formatting
            try:
                content = content.format(**args)
            except KeyError as exc:
                msg = f"Missing argument in template: {exc}"
                raise ValueError(msg) from exc
        msg_content = MessageContent(type="text", content=content)
        return [PromptMessage(role="user", content=msg_content)]


# Type to use in configuration
PromptType = Annotated[
    StaticPrompt | DynamicPrompt | FilePrompt, Field(discriminator="type")
]
