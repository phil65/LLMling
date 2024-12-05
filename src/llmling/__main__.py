"""LLMling CLI interface."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import json
import logging
from typing import Any

from py2openai import OpenAIFunctionTool  # noqa: TC002
from pydantic import BaseModel
import typer as t
import yamling

from llmling.config.manager import ConfigManager
from llmling.config.runtime import RuntimeConfig
from llmling.core.log import get_logger, setup_logging


# from rich.console import Console


# console = Console()

logger = get_logger(__name__)

cli = t.Typer(
    name="LLMling",
    help=(
        "🤖 LLMling CLI interface. Interact with resources, tools, and prompts! 🤖\n\n"
        "Check out https://github.com/phil65/llmling !"
    ),
    no_args_is_help=True,
)

# Command groups
resources_cli = t.Typer(help="Resource management commands.", no_args_is_help=True)
tools_cli = t.Typer(help="Tool management commands.", no_args_is_help=True)
prompts_cli = t.Typer(help="Prompt management commands.", no_args_is_help=True)

cli.add_typer(resources_cli, name="resource")
cli.add_typer(tools_cli, name="tool")
cli.add_typer(prompts_cli, name="prompt")

# Help texts
CONFIG_HELP = "Path to LLMling configuration file"
FORMAT_HELP = "Output format. One of: text, json, yaml"
VERBOSE_HELP = "Enable debug logging"
RESOURCE_NAME_HELP = "Name of the resource to process"
TOOL_NAME_HELP = "Name of the tool to execute"
PROMPT_NAME_HELP = "Name of the prompt to use"
ARGS_HELP = "Tool arguments in key=value format (can be specified multiple times)"

# Command options
CONFIG_CMDS = "-c", "--config"
FORMAT_CMDS = "-f", "--format"
VERBOSE_CMDS = "-v", "--verbose"


def verbose_callback(ctx: t.Context, _param: t.CallbackParam, value: bool):
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


def get_runtime(config_path: str | None = None) -> RuntimeConfig:
    """Create runtime from config."""
    if not config_path:
        from llmling import config_resources

        config_path = config_resources.TEST_CONFIG
    manager = ConfigManager.load(config_path)
    return RuntimeConfig.from_config(manager.config)


@resources_cli.command("list")
def list_resources(
    config_path: str = t.Option(None, *CONFIG_CMDS, help=CONFIG_HELP, show_default=False),
    output_format: str = t.Option("text", *FORMAT_CMDS, help=FORMAT_HELP),
    verbose: bool = t.Option(
        False, *VERBOSE_CMDS, help=VERBOSE_HELP, is_flag=True, callback=verbose_callback
    ),
):
    """List all configured resources."""
    runtime = get_runtime(config_path)
    format_models(runtime.get_resources(), output_format)


@resources_cli.command("show")
def show_resource(
    name: str = t.Argument(help=RESOURCE_NAME_HELP),
    config_path: str = t.Option(None, *CONFIG_CMDS, help=CONFIG_HELP, show_default=False),
    output_format: str = t.Option("text", *FORMAT_CMDS, help=FORMAT_HELP),
    verbose: bool = t.Option(
        False, *VERBOSE_CMDS, help=VERBOSE_HELP, is_flag=True, callback=verbose_callback
    ),
):
    """Show details of a specific resource."""
    runtime = get_runtime(config_path)
    format_models(runtime.get_resource(name), output_format)


@resources_cli.command("load")
def load_resource(
    name: str = t.Argument(help=RESOURCE_NAME_HELP),
    config_path: str = t.Option(None, *CONFIG_CMDS, help=CONFIG_HELP, show_default=False),
    verbose: bool = t.Option(
        False, *VERBOSE_CMDS, help=VERBOSE_HELP, is_flag=True, callback=verbose_callback
    ),
):
    """Load and display resource content."""
    runtime = get_runtime(config_path)

    async def _load():
        async with runtime as r:
            return await r.load_resource(name)

    print(asyncio.run(_load()))


@tools_cli.command("list")
def list_tools(
    config_path: str = t.Option(None, *CONFIG_CMDS, help=CONFIG_HELP, show_default=False),
    output_format: str = t.Option("text", *FORMAT_CMDS, help=FORMAT_HELP),
    verbose: bool = t.Option(
        False, *VERBOSE_CMDS, help=VERBOSE_HELP, is_flag=True, callback=verbose_callback
    ),
):
    """List available tools."""
    runtime = get_runtime(config_path)
    format_models(runtime.get_tools(), output_format)


@tools_cli.command("show")
def show_tool(
    name: str = t.Argument(help=TOOL_NAME_HELP),
    config_path: str = t.Option(None, *CONFIG_CMDS, help=CONFIG_HELP, show_default=False),
    output_format: str = t.Option("text", *FORMAT_CMDS, help=FORMAT_HELP),
    verbose: bool = t.Option(
        False, *VERBOSE_CMDS, help=VERBOSE_HELP, is_flag=True, callback=verbose_callback
    ),
):
    """Show tool documentation and schema."""
    runtime = get_runtime(config_path)
    format_models(runtime.get_tool(name), output_format)


@tools_cli.command("call")
def call_tool(
    name: str = t.Argument(help=TOOL_NAME_HELP),
    args: list[str] = t.Argument(None, help=ARGS_HELP),  # noqa: B008
    config_path: str = t.Option(None, *CONFIG_CMDS, help=CONFIG_HELP, show_default=False),
    verbose: bool = t.Option(
        False, *VERBOSE_CMDS, help=VERBOSE_HELP, is_flag=True, callback=verbose_callback
    ),
):
    """Execute a tool with given arguments."""
    runtime = get_runtime(config_path)
    kwargs = dict(arg.split("=", 1) for arg in (args or []))

    async def _call():
        async with runtime as r:
            return await r.execute_tool(name, **kwargs)

    print(asyncio.run(_call()))


@prompts_cli.command("list")
def list_prompts(
    config_path: str = t.Option(None, *CONFIG_CMDS, help=CONFIG_HELP, show_default=False),
    output_format: str = t.Option("text", *FORMAT_CMDS, help=FORMAT_HELP),
    verbose: bool = t.Option(
        False, *VERBOSE_CMDS, help=VERBOSE_HELP, is_flag=True, callback=verbose_callback
    ),
):
    """List available prompts."""
    runtime = get_runtime(config_path)
    format_models(runtime.get_prompts(), output_format)


@prompts_cli.command("show")
def show_prompt(
    name: str = t.Argument(help=PROMPT_NAME_HELP),
    config_path: str = t.Option(None, *CONFIG_CMDS, help=CONFIG_HELP, show_default=False),
    output_format: str = t.Option("text", *FORMAT_CMDS, help=FORMAT_HELP),
    verbose: bool = t.Option(
        False, *VERBOSE_CMDS, help=VERBOSE_HELP, is_flag=True, callback=verbose_callback
    ),
):
    """Show prompt details."""
    runtime = get_runtime(config_path)
    format_models(runtime.get_prompt(name), output_format)


if __name__ == "__main__":
    cli()
