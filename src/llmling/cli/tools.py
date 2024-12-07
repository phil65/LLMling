from __future__ import annotations

import asyncio

import typer as t

from llmling.cli.constants import (
    ARGS_HELP,
    CONFIG_CMDS,
    CONFIG_HELP,
    FORMAT_CMDS,
    FORMAT_HELP,
    TOOL_NAME_HELP,
    VERBOSE_CMDS,
    VERBOSE_HELP,
)
from llmling.cli.utils import format_output, verbose_callback
from llmling.config.runtime import RuntimeConfig


tools_cli = t.Typer(help="Tool management commands.", no_args_is_help=True)


@tools_cli.command("list")
def list_tools(
    config_path: str = t.Option(None, *CONFIG_CMDS, help=CONFIG_HELP, show_default=False),
    output_format: str = t.Option("text", *FORMAT_CMDS, help=FORMAT_HELP),
    verbose: bool = t.Option(
        False, *VERBOSE_CMDS, help=VERBOSE_HELP, is_flag=True, callback=verbose_callback
    ),
):
    """List available tools."""
    with RuntimeConfig.open_sync(config_path) as runtime:
        format_output(runtime.get_tools(), output_format)


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
    with RuntimeConfig.open_sync(config_path) as runtime:
        format_output(runtime.get_tool(name), output_format)


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
    with RuntimeConfig.open_sync(config_path) as runtime:
        kwargs = dict(arg.split("=", 1) for arg in (args or []))
        with RuntimeConfig.open_sync(config_path) as runtime:
            result = asyncio.run(runtime.execute_tool(name, **kwargs))
            print(result)
