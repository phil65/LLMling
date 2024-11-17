"""Command line interface for LLMling."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

import typer as t
import upath
from upath import UPath

from llmling.client import LLMLingClient
from llmling.core import exceptions
from llmling.core.log import setup_logging


if TYPE_CHECKING:
    import os

# Help text constants
APP_HELP = "LLMling CLI - Execute LLM tasks with streaming support"
CONFIG_PATH_HELP = "Path to YAML configuration file"
TASK_NAME_HELP = "Name of the task template to execute"
OUTPUT_FILE_HELP = "Path to save output (prints to stdout if not specified)"
STREAM_HELP = "Enable streaming output"
VERBOSE_HELP = "Enable verbose logging"
SYSTEM_PROMPT_HELP = "Optional system prompt to use"

app = t.Typer(
    help=APP_HELP,
    name="llmling",
)


@app.command()
def run(
    config_path: str = t.Argument(
        ..., help=CONFIG_PATH_HELP, exists=True, dir_okay=False, resolve_path=True
    ),
    task_name: str = t.Argument(..., help=TASK_NAME_HELP),
    output_file: str | None = t.Option(None, "--output", "-o", help=OUTPUT_FILE_HELP),
    stream: bool = t.Option(False, "--stream", "-s", help=STREAM_HELP),
    verbose: bool = t.Option(False, "--verbose", "-v", help=VERBOSE_HELP),
    system_prompt: str | None = t.Option(
        None, "--system-prompt", help=SYSTEM_PROMPT_HELP
    ),
) -> None:
    """Execute a task template with optional streaming."""
    # Setup logging
    setup_logging(level="DEBUG" if verbose else "INFO")

    try:
        # Run async execution in event loop
        asyncio.run(
            _execute_task(
                config_path=config_path,
                task_name=task_name,
                output_file=output_file,
                stream=stream,
                system_prompt=system_prompt,
            )
        )
    except exceptions.LLMLingError as exc:
        t.secho(f"Error: {exc}", fg=t.colors.RED, err=True)
        raise t.Exit(1) from exc
    except KeyboardInterrupt:
        t.secho("\nOperation cancelled by user", fg=t.colors.YELLOW)
        raise t.Exit(130) from None


async def _execute_task(
    config_path: str | os.PathLike[str],
    task_name: str,
    output_file: str | None,
    stream: bool,
    system_prompt: str | None,
) -> None:
    """Execute task asynchronously."""
    async with LLMLingClient(config_path) as client:
        try:
            if stream:
                await _handle_streaming(
                    client=client,
                    task_name=task_name,
                    output_file=output_file,
                    system_prompt=system_prompt,
                )
            else:
                await _handle_single(
                    client=client,
                    task_name=task_name,
                    output_file=output_file,
                    system_prompt=system_prompt,
                )
        except exceptions.LLMLingError as exc:
            msg = f"Task execution failed: {exc}"
            raise exceptions.LLMLingError(msg) from exc


async def _handle_streaming(
    client: LLMLingClient,
    task_name: str,
    output_file: str | None,
    system_prompt: str | None,
) -> None:
    """Handle streaming execution."""
    # Prepare output file if specified
    output_stream = (
        upath.UPath(output_file).open("w", encoding="utf-8")
        if output_file
        else sys.stdout
    )

    try:
        t.secho("Starting streaming execution...", fg=t.colors.BLUE)
        async for chunk in await client.execute(
            task_name,
            stream=True,
            system_prompt=system_prompt,
        ):
            print(chunk.content, end="", flush=True, file=output_stream)

    finally:
        if output_file:
            output_stream.close()


async def _handle_single(
    client: LLMLingClient,
    task_name: str,
    output_file: str | None,
    system_prompt: str | None,
) -> None:
    """Handle non-streaming execution."""
    result = await client.execute(
        task_name,
        system_prompt=system_prompt,
    )

    # Handle output
    if output_file:
        UPath(output_file).write_text(result.content)
        t.secho(
            f"Output saved to {output_file}",
            fg=t.colors.GREEN,
        )
    else:
        print(result.content)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
