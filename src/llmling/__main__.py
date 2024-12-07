"""LLMling CLI interface."""

from __future__ import annotations

import typer as t

from llmling.cli.prompts import prompts_cli
from llmling.cli.resources import resources_cli
from llmling.cli.tools import tools_cli


HELP = """
🤖 LLMling CLI interface. Interact with resources, tools, and prompts! 🤖

Check out https://github.com/phil65/llmling !
"""

cli = t.Typer(name="LLMling", help=HELP, no_args_is_help=True)
cli.add_typer(resources_cli, name="resource")
cli.add_typer(tools_cli, name="tool")
cli.add_typer(prompts_cli, name="prompt")


if __name__ == "__main__":
    cli(["resource", "list"])
