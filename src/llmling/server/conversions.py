"""Conversions between internal and MCP types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mcp.types


if TYPE_CHECKING:
    from llmling.processors.base import ProcessorConfig
    from llmling.prompts.models import (
        ExtendedPromptArgument,
        Prompt as InternalPrompt,
        PromptMessage,
    )
    from llmling.resources.models import LoadedResource
    from llmling.tools.base import LLMCallableTool


def to_mcp_tool(tool: LLMCallableTool) -> mcp.types.Tool:
    """Convert internal Tool to MCP Tool."""
    schema = tool.get_schema()
    return mcp.types.Tool(
        name=schema["function"]["name"],
        description=schema["function"]["description"],
        inputSchema=schema["function"]["parameters"],
    )


def to_mcp_resource(resource: LoadedResource) -> mcp.types.Resource:
    """Convert LoadedResource to MCP Resource."""
    return mcp.types.Resource(
        uri=mcp.types.AnyUrl(resource.metadata.uri),
        name=resource.metadata.name or "",
        description=resource.metadata.description,
        mimeType=resource.metadata.mime_type,
    )


def to_mcp_message(msg: PromptMessage) -> mcp.types.PromptMessage:
    """Convert internal PromptMessage to MCP PromptMessage."""
    role: mcp.types.Role = "assistant" if msg.role == "assistant" else "user"
    return mcp.types.PromptMessage(
        role=role,
        content=mcp.types.TextContent(
            type="text",
            text=msg.get_text_content(),
        ),
    )


def to_mcp_capability(proc_config: ProcessorConfig) -> dict[str, Any]:
    """Convert to MCP capability format."""
    return {
        "name": proc_config.name,
        "type": proc_config.type,
        "description": proc_config.description,
        "mimeTypes": proc_config.supported_mime_types,
        "maxInputSize": proc_config.max_input_size,
        "streaming": proc_config.streaming,
    }


def to_mcp_argument(prompt_arg: ExtendedPromptArgument) -> mcp.types.PromptArgument:
    """Convert to MCP PromptArgument."""
    return mcp.types.PromptArgument(
        name=prompt_arg.name,
        description=prompt_arg.description,
        required=prompt_arg.required,
    )


def to_mcp_prompt(prompt: InternalPrompt) -> mcp.types.Prompt:
    """Convert to MCP Prompt."""
    return mcp.types.Prompt(
        name=prompt.name,
        description=prompt.description,
        arguments=[to_mcp_argument(arg) for arg in prompt.arguments],
    )
