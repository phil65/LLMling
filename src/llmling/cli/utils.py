from __future__ import annotations

from collections.abc import Sequence
import json
import logging
from typing import Any

from py2openai import OpenAIFunctionTool  # noqa: TC002
from pydantic import BaseModel
import typer as t  # noqa: TC002
import yamling

from llmling.core.log import setup_logging


# from rich.console import Console
# console = Console()


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


def to_model(obj: Any) -> BaseModel | Sequence[BaseModel]:
    """Convert objects to BaseModel representations if needed."""
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
            return [to_model(item) for item in obj]
        case BaseModel():
            return obj
        case _:
            msg = f"Cannot convert type {type(obj)} to model"
            raise TypeError(msg)


def format_models(result: Any, output_format: str = "text") -> None:
    """Format and print models.

    Args:
        result: BaseModel, Sequence[BaseModel], or LLMCallableTool(s)
               Will be converted to BaseModel representation if needed.
        output_format: One of: text, json, yaml

    Raises:
        TypeError: If result cannot be converted to a BaseModel
        ValueError: If output_format is invalid
    """
    model = to_model(result)  # converts to BaseModel | Sequence[BaseModel]
    match output_format:
        case "json":
            if isinstance(model, Sequence):
                print(json.dumps([m.model_dump() for m in model], indent=2))
            else:
                print(model.model_dump_json(indent=2))
        case "yaml":
            if isinstance(model, Sequence):
                print(yamling.dump_yaml([m.model_dump() for m in model]))
            else:
                print(yamling.dump_yaml(model.model_dump()))
        case "text":
            print(model)
        case _:
            msg = f"Unknown format: {output_format}"
            raise ValueError(msg)
