from __future__ import annotations

import typer as t

from llmling.cli.constants import output_format_opt
from llmling.cli.utils import OutputFormat, format_output, get_command_help
from llmling.config.store import config_store


help_text = get_command_help("Config file management commands.")
config_cli = t.Typer(help=help_text, no_args_is_help=True)


@config_cli.command("add")
def add_config(
    name: str = t.Argument(help="Name for the config"),
    path: str = t.Argument(help="Path to config file"),
) -> None:
    """Add a new named config (llmling config add <identifier> <path>).

    Paths can be UPath protocol paths (protocol://path) or local file paths.
    """
    try:
        config_store.add_config(name, path)
        print(f"Added config '{name}' -> {path}")
    except Exception as exc:
        print(f"Failed to add config: {exc}")
        raise t.Exit(1) from exc


@config_cli.command("remove")
def remove_config(
    name: str = t.Argument(help="Name of config to remove"),
) -> None:
    """Remove a named config."""
    try:
        config_store.remove_config(name)
        print(f"Removed config '{name}'")
    except Exception as exc:
        print(f"Failed to remove config: {exc}")
        raise t.Exit(1) from exc


@config_cli.command("set")
def set_active(
    name: str = t.Argument(help="Name of config to set as active"),
) -> None:
    """Set the active config."""
    try:
        config_store.set_active(name)
        print(f"Set '{name}' as active config")
    except Exception as exc:
        print(f"Failed to set active config: {exc}")
        raise t.Exit(1) from exc


@config_cli.command("list")
def list_configs(
    output_format: OutputFormat = output_format_opt,
) -> None:
    """List all registered configs."""
    try:
        configs = config_store.list_configs()
        active = config_store.get_active()

        # Create formatted output
        result = {
            "configs": [
                {
                    "name": name,
                    "path": uri,
                    "active": active and name == active.name,
                }
                for name, uri in configs
            ],
        }
        format_output(result, output_format)
    except Exception as exc:
        print(f"Failed to list configs: {exc}")
        raise t.Exit(1) from exc


@config_cli.command("init")
def init_config(
    output: str = t.Argument(help="Path to write configuration file"),
    interactive: bool = t.Option(
        False,
        "--interactive/--no-interactive",
        help="Use interactive configuration wizard",
    ),
):
    """Initialize a new configuration file with basic settings.

    Creates a new configuration file at the specified path. Use --interactive
    for a guided setup process.
    """
    from promptantic import ModelGenerator

    from llmling import Config

    if interactive:
        generator = ModelGenerator()
        config = generator.populate(Config)
        config.save(output)
        print(f"Created configuration file: {output}")
    else:
        import shutil

        from llmling import config_resources

        shutil.copy2(config_resources.BASIC_CONFIG, output)
        print(f"Created configuration file: {output}")
        print("\nTry these commands:")
        print("  llmling resource list")
        print("  llmling tool call open_url url=https://github.com")
        print("  llmling prompt show greet")


@config_cli.command("show")
def show_config(
    name: str | None = t.Argument(
        None,
        help="Name of config to show (shows active if not provided)",
    ),
    output_format: OutputFormat = output_format_opt,
    resolve: bool = t.Option(
        False,
        "--resolve/--no-resolve",
        help="Show resolved configuration with expanded toolsets",
    ),
) -> None:
    """Show current configuration.

    With --resolve (default), shows the fully resolved configuration
    including expanded toolsets and loaded resources.
    """
    from llmling.config.models import Config
    from llmling.config.runtime import RuntimeConfig

    try:
        # Get config path to show
        if name is None:
            if active := config_store.get_active():
                path = active.path
            else:
                print("No active config set")
                raise t.Exit(1)  # noqa: TRY301
        else:
            path = config_store.get_config(name)

        if not resolve:
            # Just show the raw config file
            config = Config.from_file(path)
            format_output(config, output_format)
            return

        # Show resolved configuration
        with RuntimeConfig.open_sync(path) as runtime:
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

    except Exception as exc:
        print(f"Failed to show config: {exc}")
        raise t.Exit(1) from exc
