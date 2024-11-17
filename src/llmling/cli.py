"""Command line interface for LLMling."""

from __future__ import annotations

import asyncio
import enum
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from upath import UPath

from llmling.config.loading import load_config
from llmling.context import default_registry as context_registry
from llmling.llm.registry import default_registry as llm_registry
from llmling.processors.registry import ProcessorRegistry
from llmling.task.executor import TaskExecutor
from llmling.task.manager import TaskManager


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class ExitCode(enum.IntEnum):
    """Standard exit codes for the CLI."""

    SUCCESS = 0
    ERROR = 1  # General errors (config invalid, execution failed, etc)
    USAGE = 2  # CLI usage errors (missing args, invalid options, etc)


console = Console()


@click.group()
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration file",
    required=True,
)
@click.pass_context
def cli(ctx: click.Context, config: str) -> None:
    """LLMling - LLM interaction management system."""
    ctx.ensure_object(dict)

    config_path = UPath(config)
    if not config_path.exists():
        ctx.exit(ExitCode.USAGE)
        return

    try:
        ctx.obj["config"] = load_config(config_path)
    except Exception as exc:
        error_msg = f"[red]Configuration Error:[/red] {exc}"
        console.print(error_msg)
        ctx.exit(ExitCode.ERROR)

    # Initialize components
    processor_registry = ProcessorRegistry()
    executor = TaskExecutor(
        context_registry=context_registry,
        processor_registry=processor_registry,
        provider_registry=llm_registry,
    )
    ctx.obj["manager"] = TaskManager(ctx.obj["config"], executor)


async def execute_task(
    manager: TaskManager,
    template: str,
    system_prompt: str | None,
    stream: bool,
) -> AsyncGenerator[str, None]:
    """Execute a task and yield results."""
    try:
        if stream:
            async for result in manager.execute_template_stream(
                template,
                system_prompt=system_prompt,
            ):
                yield result.content
        else:
            result = await manager.execute_template(
                template,
                system_prompt=system_prompt,
            )
            yield result.content
    except Exception as exc:
        error_msg = f"[red]Error:[/red] {exc}"
        yield error_msg
        raise Exception(ExitCode.ERROR)


def run_async(coro):
    """Run an async function in the current loop or create a new one."""
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # We're in an async context - create a task
            return loop.create_task(coro)
    except RuntimeError:
        # No loop running - create one and run it
        return asyncio.run(coro)


@cli.command()
@click.argument("template")
@click.option(
    "--system-prompt",
    help="Optional system prompt",
)
@click.option(
    "--stream",
    is_flag=True,
    help="Stream results",
)
@click.pass_context
def execute(
    ctx: click.Context,
    template: str,
    system_prompt: str | None,
    stream: bool,
) -> None:
    """Execute a task template."""
    manager: TaskManager = ctx.obj["manager"]

    async def run_task() -> None:
        async for content in execute_task(manager, template, system_prompt, stream):
            console.print(content, end="" if stream else "\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress_task = progress.add_task("Executing task...", total=None)

        run_async(run_task())

        progress.update(progress_task, completed=True)


def get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop safely."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


@cli.command()
@click.pass_context
def list_templates(ctx: click.Context) -> None:
    """List available task templates."""
    config = ctx.obj["config"]

    console.print("\n[bold]Available Task Templates:[/bold]\n")

    for name, template in config.task_templates.items():
        console.print(f"[green]{name}[/green]")
        console.print(f"  Provider: {template.provider}")
        console.print(f"  Context: {template.context}")
        if template.settings:
            console.print("  Settings:")
            for key, value in template.settings.model_dump().items():
                if value is not None:
                    console.print(f"    {key}: {value}")
        console.print()


@cli.command()
@click.pass_context
def validate(ctx: click.Context) -> None:
    """Validate configuration."""
    config = ctx.obj["config"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        validation_task = progress.add_task("Validating configuration...", total=None)

        try:
            # Validation happens during loading, so if we got here, it's valid
            progress.update(validation_task, completed=True)
            console.print("[green]Configuration is valid![/green]")

            # Print summary
            console.print("\n[bold]Configuration Summary:[/bold]")
            console.print(f"Providers: {len(config.llm_providers)}")
            console.print(f"Contexts: {len(config.contexts)}")
            console.print(f"Templates: {len(config.task_templates)}")

        except Exception as exc:
            error_msg = f"[red]Validation Error:[/red] {exc}"
            console.print(error_msg)
            ctx.exit(ExitCode.ERROR)


if __name__ == "__main__":
    cli()
