from __future__ import annotations

import json
import logging
from typing import Any

from py2openai import OpenAIFunctionTool  # noqa: TC002
from pydantic import BaseModel
from rich.console import Console
import typer as t  # noqa: TC002
import yamling

from llmling.core.log import setup_logging


console = Console()


def verbose_callback(ctx: t.Context, _param: t.CallbackParam, value: bool) -> bool:
    """Handle verbose flag."""
    if value:
        setup_logging(level=logging.DEBUG)
    return value


class ToolDisplay(BaseModel):
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
                system_prompt=obj.system_prompt,
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


def format_output(result: Any, output_format: str = "text") -> None:
    """Format and print data in the requested format.

    Args:
        result: Object to format (BaseModel, dict, or sequence)
        output_format: One of: text, json, yaml
    """
    data = prepare_for_output(result)

    match output_format:
        case "json":
            if isinstance(data, BaseModel):
                print(data.model_dump_json(indent=2))
            else:
                print(json.dumps(data, indent=2))
        case "yaml":
            if isinstance(data, BaseModel):
                print(yamling.dump_yaml(data.model_dump()))
            else:
                print(yamling.dump_yaml(data))
        case "text":
            console.print(data)
        case _:
            msg = f"Unknown format: {output_format}"
            raise ValueError(msg)
