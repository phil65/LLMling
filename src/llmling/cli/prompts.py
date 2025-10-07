from __future__ import annotations

import typer as t

from llmling.cli.constants import (
    PROMPT_NAME_HELP,
    config_file_opt,
    output_format_opt,
    verbose_opt,
)
from llmling.cli.utils import OutputFormat, format_output, get_command_help


help_cmd = get_command_help("Prompt management commands.")
prompts_cli = t.Typer(help=help_cmd, no_args_is_help=True)


@prompts_cli.command("list")
def list_prompts(
    config_path: str = config_file_opt,
    output_format: OutputFormat = output_format_opt,
    verbose: bool = verbose_opt,
):
    """List available prompts."""
    from llmling.config.runtime import RuntimeConfig

    with RuntimeConfig.open_sync(config_path) as runtime:
        format_output(runtime.get_prompts(), output_format)


@prompts_cli.command("show")
def show_prompt(
    config_path: str = config_file_opt,
    name: str = t.Argument(help=PROMPT_NAME_HELP),
    output_format: OutputFormat = output_format_opt,
    verbose: bool = verbose_opt,
):
    """Show prompt details."""
    from llmling.config.runtime import RuntimeConfig

    with RuntimeConfig.open_sync(config_path) as runtime:
        format_output(runtime.get_prompt(name), output_format)
