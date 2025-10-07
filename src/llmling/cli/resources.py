from __future__ import annotations

import asyncio

import typer as t

from llmling.cli.constants import (
    RESOURCE_NAME_HELP,
    config_file_opt,
    output_format_opt,
    verbose_opt,
)
from llmling.cli.utils import OutputFormat, format_output, get_command_help


help_cmd = get_command_help("Resource management commands.")
resources_cli = t.Typer(help=help_cmd, no_args_is_help=True)


@resources_cli.command("list")
def list_resources(
    config_path: str = config_file_opt,
    output_format: OutputFormat = output_format_opt,
    verbose: bool = verbose_opt,
):
    """List all configured resources."""
    from llmling.config.runtime import RuntimeConfig

    with RuntimeConfig.open_sync(config_path) as runtime:
        format_output(runtime.get_resources(), output_format)


@resources_cli.command("show")
def show_resource(
    config_path: str = config_file_opt,
    name: str = t.Argument(help=RESOURCE_NAME_HELP),
    output_format: OutputFormat = output_format_opt,
    verbose: bool = verbose_opt,
):
    """Show details of a specific resource."""
    from llmling.config.runtime import RuntimeConfig

    with RuntimeConfig.open_sync(config_path) as runtime:
        format_output(runtime.get_resource(name), output_format)


@resources_cli.command("load")
def load_resource(
    config_path: str = config_file_opt,
    name: str = t.Argument(help=RESOURCE_NAME_HELP),
    verbose: bool = verbose_opt,
):
    """Load and display resource content."""
    from llmling.config.runtime import RuntimeConfig

    with RuntimeConfig.open_sync(config_path) as runtime:

        async def _load():
            async with runtime as r:
                return await r.load_resource(name)

        print(asyncio.run(_load()))
