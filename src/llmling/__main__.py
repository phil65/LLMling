"""LLMling CLI interface."""

from __future__ import annotations

import typer as t

from llmling.cli.config import config_cli
from llmling.cli.prompts import prompts_cli
from llmling.cli.resources import resources_cli
from llmling.cli.tools import tools_cli


HELP = """
🤖 LLMling CLI interface. Interact with resources, tools, and prompts! 🤖

Check out https://github.com/phil65/llmling !
"""

cli = t.Typer(name="LLMling", help=HELP, no_args_is_help=True)
cli.add_typer(config_cli, name="config")
cli.add_typer(resources_cli, name="resource")
cli.add_typer(tools_cli, name="tool")
cli.add_typer(prompts_cli, name="prompt")
# try:
#     from mcp_server_llmling.cli import cli as server_cli
#     cli.add_typer(server_cli, name="server")
# except ImportError:
#     server_cli = t.Typer(help="MCP server commands (not installed)")

#     @server_cli.callback()
#     def server_not_installed():
#         """MCP server functionality (not installed)."""
#         msg = (
#             "Server commands require the mcp-server-llmling package.\n"
#             "Install with: pip install mcp-server-llmling"
#         )
#         raise t.Exit(msg)

#     cli.add_typer(server_cli, name="server")


if __name__ == "__main__":
    cli(["resource", "list"])
