from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import typer as t

from llmling.config.store import config_store
from llmling.core.log import setup_logging


if TYPE_CHECKING:
    import click
    import click.shell_completion


# Help texts
CONFIG_HELP = "Path to config file or name of stored config"
OUTPUT_FORMAT_HELP = "Output format. One of: text, json, yaml"
VERBOSE_HELP = "Enable debug logging"
FORMAT_HELP = "Output format. One of: text, json, yaml"
RESOURCE_NAME_HELP = "Name of the resource to process"
TOOL_NAME_HELP = "Name of the tool to execute"
PROMPT_NAME_HELP = "Name of the prompt to use"
ARGS_HELP = "Tool arguments in key=value format (can be specified multiple times)"
# Command options
OUTPUT_FORMAT_CMDS = "-o", "--output-format"
CONFIG_CMDS = "-c", "--config"
FORMAT_CMDS = "-f", "--format"
VERBOSE_CMDS = "-v", "--verbose"


def complete_config_names(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[str] | list[click.shell_completion.CompletionItem]:
    """Complete stored config names."""
    names = [name for name, _ in config_store.list_configs()]
    return [name for name in names if name.startswith(incomplete)]


def complete_output_formats(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[str]:
    """Complete output format options."""
    formats = ["text", "json", "yaml"]
    return [f for f in formats if f.startswith(incomplete)]


def verbose_callback(ctx: t.Context, _param: t.CallbackParam, value: bool) -> bool:
    """Handle verbose flag."""
    if value:
        setup_logging(level=logging.DEBUG)
    return value


config_file_opt = t.Option(
    None, "-c", "--config", shell_complete=complete_config_names, help=CONFIG_HELP
)

output_format_opt = t.Option(
    "text",
    *OUTPUT_FORMAT_CMDS,
    help=OUTPUT_FORMAT_HELP,
    shell_complete=complete_output_formats,
)
verbose_opt = t.Option(
    False, *VERBOSE_CMDS, help=VERBOSE_HELP, is_flag=True, callback=verbose_callback
)
