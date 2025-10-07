from __future__ import annotations

import asyncio

import typer as t

from llmling.cli.constants import (
    ARGS_HELP,
    TOOL_NAME_HELP,
    config_file_opt,
    output_format_opt,
    verbose_opt,
)
from llmling.cli.utils import OutputFormat, format_output, get_command_help


help_cmd = get_command_help("Tool management commands.")
tools_cli = t.Typer(help=help_cmd, no_args_is_help=True)


@tools_cli.command("list")
def list_tools(
    config_path: str = config_file_opt,
    output_format: OutputFormat = output_format_opt,
    verbose: bool = verbose_opt,
):
    """List available tools."""
    from llmling.config.runtime import RuntimeConfig

    with RuntimeConfig.open_sync(config_path) as runtime:
        format_output(runtime.get_tools(), output_format)


@tools_cli.command("show")
def show_tool(
    config_path: str = config_file_opt,
    name: str = t.Argument(help=TOOL_NAME_HELP),
    output_format: OutputFormat = output_format_opt,
    verbose: bool = verbose_opt,
):
    """Show tool documentation and schema."""
    from llmling.config.runtime import RuntimeConfig

    with RuntimeConfig.open_sync(config_path) as runtime:
        format_output(runtime.get_tool(name), output_format)


@tools_cli.command("call")
def call_tool(
    config_path: str = config_file_opt,
    name: str = t.Argument(help=TOOL_NAME_HELP),
    args: list[str] = t.Argument(None, help=ARGS_HELP),  # noqa: B008
    verbose: bool = verbose_opt,
):
    """Execute a tool with given arguments."""
    from llmling.config.runtime import RuntimeConfig

    kwargs = dict(arg.split("=", 1) for arg in (args or []))
    with RuntimeConfig.open_sync(config_path) as runtime:
        result = asyncio.run(runtime.execute_tool(name, **kwargs))
        print(result)
