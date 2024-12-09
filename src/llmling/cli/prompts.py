from __future__ import annotations

import typer as t

from llmling.cli.constants import (
    FORMAT_CMDS,
    FORMAT_HELP,
    PROMPT_NAME_HELP,
    VERBOSE_CMDS,
    VERBOSE_HELP,
    config_file_arg,
)
from llmling.cli.utils import format_output, verbose_callback


prompts_cli = t.Typer(help="Prompt management commands.", no_args_is_help=True)


@prompts_cli.command("list")
def list_prompts(
    config_path: str = config_file_arg,
    output_format: str = t.Option("text", *FORMAT_CMDS, help=FORMAT_HELP),
    verbose: bool = t.Option(
        False, *VERBOSE_CMDS, help=VERBOSE_HELP, is_flag=True, callback=verbose_callback
    ),
):
    """List available prompts."""
    from llmling.config.runtime import RuntimeConfig

    with RuntimeConfig.open_sync(config_path) as runtime:
        format_output(runtime.get_prompts(), output_format)


@prompts_cli.command("show")
def show_prompt(
    config_path: str = config_file_arg,
    name: str = t.Argument(help=PROMPT_NAME_HELP),
    output_format: str = t.Option("text", *FORMAT_CMDS, help=FORMAT_HELP),
    verbose: bool = t.Option(
        False, *VERBOSE_CMDS, help=VERBOSE_HELP, is_flag=True, callback=verbose_callback
    ),
):
    """Show prompt details."""
    from llmling.config.runtime import RuntimeConfig

    with RuntimeConfig.open_sync(config_path) as runtime:
        format_output(runtime.get_prompt(name), output_format)
