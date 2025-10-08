from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel
from schemez import OpenAIFunctionTool, Schema

from llmling.config.store import config_store


if TYPE_CHECKING:
    import os

OutputFormat = Literal["text", "json", "yaml"]

NO_CONFIG_MESSAGE = """
No configuration specified. To fix this:

1. Use -c/--config to specify a config file:
   llmling -c path/to/config.yml resource list

2. Set an active config:
   llmling config add myconfig path/to/config.yml
   llmling config set myconfig

3. Create a new config:
   llmling config init path/to/config.yml
""".strip()


class ToolDisplay(Schema):
    """Display representation of a LLMCallableTool."""

    name: str
    description: str
    function_schema: OpenAIFunctionTool
    system_prompt: str | None = None
    import_path: str | None = None


def prepare_for_output(obj: Any) -> BaseModel | dict[str, Any] | list[Any]:
    """Prepare object for output formatting.

    Converts LLMCallableTools to display models, keeps dicts as-is,
    and handles sequences.
    """
    from llmling.tools.base import LLMCallableTool

    match obj:
        case LLMCallableTool():
            return ToolDisplay(
                name=obj.name,
                description=obj.description,
                function_schema=obj.get_schema(),
                # system_prompt=obj.system_prompt,
                import_path=obj.import_path,
            )
        case list() | tuple():
            return [prepare_for_output(item) for item in obj]
        case BaseModel():
            return obj
        case dict():
            return obj
        case _:
            msg = f"Cannot format type {type(obj)}"
            raise TypeError(msg)


def format_output(
    result: Any,
    output_format: OutputFormat = "text",
) -> None:
    """Format and print data in the requested format.

    Args:
        result: Object to format (BaseModel, dict, or sequence)
        output_format: One of: text, json, yaml
    """
    data = prepare_for_output(result)
    from rich.console import Console

    console = Console()
    match output_format:
        case "json":
            if isinstance(data, BaseModel):
                print(data.model_dump_json(indent=2))
            else:
                print(json.dumps(data, indent=2))
        case "yaml":
            import yamling

            if isinstance(data, BaseModel):
                print(yamling.dump_yaml(data.model_dump()))
            else:
                print(yamling.dump_yaml(data))
        case "text":
            console.print(data)
        case _:
            msg = f"Unknown format: {output_format}"
            raise ValueError(msg)


def resolve_config_path(config: str | os.PathLike[str] | None) -> str:
    """Resolve config path from name or direct path."""
    if not config:
        if active := config_store.get_active():
            return active.path
        raise ValueError(NO_CONFIG_MESSAGE)

    try:
        # First try as stored config name
        return config_store.get_config(str(config))
    except KeyError:
        # If not found, treat as direct path
        return str(config)


def get_command_help(base_help: str) -> str:
    """Get command help text with active config information."""
    if active := config_store.get_active():
        return f"{base_help}\n\n(Using config: {active})"
    return f"{base_help}\n\n(No active config set)"
