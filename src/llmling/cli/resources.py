from __future__ import annotations

import asyncio

import typer as t

from llmling.cli.constants import (
    CONFIG_CMDS,
    CONFIG_HELP,
    FORMAT_CMDS,
    FORMAT_HELP,
    RESOURCE_NAME_HELP,
    VERBOSE_CMDS,
    VERBOSE_HELP,
)
from llmling.cli.utils import format_output, verbose_callback
from llmling.config.runtime import RuntimeConfig


resources_cli = t.Typer(help="Resource management commands.", no_args_is_help=True)


@resources_cli.command("list")
def list_resources(
    config_path: str = t.Option(None, *CONFIG_CMDS, help=CONFIG_HELP, show_default=False),
    output_format: str = t.Option("text", *FORMAT_CMDS, help=FORMAT_HELP),
    verbose: bool = t.Option(
        False, *VERBOSE_CMDS, help=VERBOSE_HELP, is_flag=True, callback=verbose_callback
    ),
):
    """List all configured resources."""
    with RuntimeConfig.open_sync(config_path) as runtime:
        format_output(runtime.get_resources(), output_format)


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
    with RuntimeConfig.open_sync(config_path) as runtime:
        format_output(runtime.get_resource(name), output_format)


@resources_cli.command("load")
def load_resource(
    name: str = t.Argument(help=RESOURCE_NAME_HELP),
    config_path: str = t.Option(None, *CONFIG_CMDS, help=CONFIG_HELP, show_default=False),
    verbose: bool = t.Option(
        False, *VERBOSE_CMDS, help=VERBOSE_HELP, is_flag=True, callback=verbose_callback
    ),
):
    """Load and display resource content."""
    with RuntimeConfig.open_sync(config_path) as runtime:

        async def _load():
            async with runtime as r:
                return await r.load_resource(name)

        print(asyncio.run(_load()))
