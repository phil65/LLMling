"""Conversions between internal and MCP types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp import types


if TYPE_CHECKING:
    from llmling.processors.base import ProcessorConfig
    from llmling.prompts.models import (
        ExtendedPromptArgument,
        Prompt as InternalPrompt,
        PromptMessage,
    )
    from llmling.resources.models import LoadedResource
    from llmling.tools.base import LLMCallableTool


def to_mcp_tool(tool: LLMCallableTool) -> types.Tool:
    """Convert internal Tool to MCP Tool."""
    schema = tool.get_schema()
    return types.Tool(
        name=schema["function"]["name"],
        description=schema["function"]["description"],
        inputSchema=schema["function"]["parameters"],
    )


def to_mcp_resource(resource: LoadedResource) -> types.Resource:
    """Convert LoadedResource to MCP Resource."""
    return types.Resource(
        uri=to_mcp_uri(resource.metadata.uri),
        name=resource.metadata.name or "",
        description=resource.metadata.description,
        mimeType=resource.metadata.mime_type,
    )


def to_mcp_message(msg: PromptMessage) -> types.PromptMessage:
    """Convert internal PromptMessage to MCP PromptMessage."""
    role: types.Role = "assistant" if msg.role == "assistant" else "user"
    return types.PromptMessage(
        role=role,
        content=types.TextContent(
            type="text",
            text=msg.get_text_content(),
        ),
    )


def to_mcp_capability(proc_config: ProcessorConfig) -> dict[str, Any]:
    """Convert to MCP capability format."""
    capability = {
        "name": proc_config.name,
        "type": proc_config.type,
        "description": proc_config.description,
        "mimeTypes": proc_config.supported_mime_types,
        "maxInputSize": proc_config.max_input_size,
        "streaming": proc_config.streaming,
    }
    return {k: v for k, v in capability.items() if v is not None}


def to_mcp_argument(prompt_arg: ExtendedPromptArgument) -> types.PromptArgument:
    """Convert to MCP PromptArgument."""
    return types.PromptArgument(
        name=prompt_arg.name,
        description=prompt_arg.description,
        required=prompt_arg.required,
    )


def to_mcp_prompt(prompt: InternalPrompt) -> types.Prompt:
    """Convert to MCP Prompt."""
    return types.Prompt(
        name=prompt.name,
        description=prompt.description,
        arguments=[to_mcp_argument(arg) for arg in prompt.arguments],
    )


def to_mcp_uri(uri: str) -> types.AnyUrl:
    """Convert internal URI to MCP-compatible AnyUrl."""
    try:
        scheme = uri.split("://", 1)[0] if "://" in uri else ""

        match scheme:
            case "http" | "https":
                return types.AnyUrl(uri)
            case "file":
                path = uri.split(":", 1)[1].lstrip("/")
                return types.AnyUrl(f"file://localhost/{path}")
            case _:
                name = uri.split("://", 1)[1]
                return types.AnyUrl(f"resource://local/{name}")
    except Exception as exc:
        msg = f"Failed to convert URI {uri!r} to MCP format"
        raise ValueError(msg) from exc
