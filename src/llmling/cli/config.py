import typer as t

from llmling.cli.constants import CONFIG_HELP, FORMAT_CMDS, FORMAT_HELP
from llmling.cli.utils import format_output
from llmling.config.models import Config
from llmling.config.runtime import RuntimeConfig


config_cli = t.Typer(help="Config file management commands.", no_args_is_help=True)


@config_cli.command("init")
def init_config(
    output: str = t.Argument(help="Path to write configuration file"),
):
    """Initialize a new configuration file with basic settings.

    Creates a new configuration file at the specified path using the basic template.
    The basic template includes a text resource, a browser tool, and example prompts
    that work out of the box.
    """
    import shutil

    from llmling import config_resources

    # Copy our basic template to the output path
    shutil.copy2(config_resources.BASIC_CONFIG, output)
    print(f"Created configuration file: {output}")
    print("\nTry these commands:")
    print("  llmling resource list")
    print("  llmling tool call open_url url=https://github.com")
    print("  llmling prompt show greet")
    print("  llmling prompt show get_user")


@config_cli.command("show")
def show_config(
    config_path: str = t.Argument(help=CONFIG_HELP),
    output_format: str = t.Option("text", *FORMAT_CMDS, help=FORMAT_HELP),
    resolve: bool = t.Option(
        True,
        "--resolve/--no-resolve",
        help="Show resolved configuration with expanded toolsets",
    ),
):
    """Show current configuration.

    With --resolve (default), shows the fully resolved configuration
    including expanded toolsets and loaded resources.
    """
    if not resolve:
        # Just show the raw config file
        config = Config.from_file(config_path)
        format_output(config, output_format)
        return

    with RuntimeConfig.open_sync(config_path) as runtime:
        # Create resolved view of configuration
        resolved = {
            "version": runtime.original_config.version,
            "global_settings": runtime.original_config.global_settings.model_dump(),
            "resources": {
                resource.uri: resource.model_dump()
                for resource in runtime.get_resources()
                if resource.uri  # ensure we have a name
            },
            "tools": {
                tool.name: {
                    "description": tool.description,
                    "import_path": tool.import_path,
                    "schema": tool.get_schema(),
                }
                for tool in runtime.get_tools()
            },
            "prompts": {
                prompt.name: prompt.model_dump()
                for prompt in runtime.get_prompts()
                if prompt.name  # ensure we have a name
            },
        }
        format_output(resolved, output_format)
