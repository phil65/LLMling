"""LLMling CLI interface."""

from __future__ import annotations

import typer as t

from llmling.cli.config import config_cli
from llmling.cli.prompts import prompts_cli
from llmling.cli.resources import resources_cli
from llmling.cli.tools import tools_cli
from llmling.config.store import ConfigStore


BASE_HELP = """
🤖 LLMling CLI interface. Interact with resources, tools, and prompts! 🤖

{config_info}

Check out https://github.com/phil65/llmling !
"""


def get_help_text() -> str:
    """Get help text with active config information."""
    try:
        if active := ConfigStore().get_active():
            return BASE_HELP.format(config_info=f"Active config: {active}")
    except Exception:  # noqa: BLE001
        pass
    return BASE_HELP.format(config_info="No active config set")


MISSING_SERVER = """
Server commands require the mcp-server-llmling package.
Install with: pip install mcp-server-llmling
"""

MISSING_AGENT = """
Agent commands require the llmling-agent package.
Install with: pip install llmling-agent
"""

cli = t.Typer(name="LLMling", help=get_help_text(), no_args_is_help=True)
cli.add_typer(config_cli, name="config")
cli.add_typer(resources_cli, name="resource")
cli.add_typer(tools_cli, name="tool")
cli.add_typer(prompts_cli, name="prompt")
try:
    from mcp_server_llmling.__main__ import cli as server_cli

    cli.add_typer(server_cli, name="server", help="MCP server commands")
except ImportError:
    server_cli = t.Typer(help="MCP server commands (not installed)")

    @server_cli.callback()
    def server_not_installed():
        """MCP server functionality (not installed)."""
        print(MISSING_SERVER)
        raise t.Exit

    cli.add_typer(server_cli, name="server")

try:
    from llmling_agent.__main__ import cli as agent_cli

    cli.add_typer(agent_cli, name="agent", help="Agent commands")
except ImportError:
    agent_cli = t.Typer(help="Agent commands (not installed)")

    @agent_cli.callback()
    def agent_not_installed():
        """Agent functionality (not installed)."""
        print(MISSING_AGENT)
        raise t.Exit

    cli.add_typer(agent_cli, name="agent")


if __name__ == "__main__":
    cli(["resource", "list"])
