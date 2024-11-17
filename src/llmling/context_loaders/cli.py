"""CLI context loader."""

from __future__ import annotations

import asyncio
import shlex
from typing import TYPE_CHECKING

from llmling.context_loaders.base import ContextLoader, LoadedContext, LoaderError
from llmling.processors import ProcessorRegistry, process_context


if TYPE_CHECKING:
    from llmling.config import CLIContext, Context


class CLILoadError(LoaderError):
    """Raised when CLI command execution or processing fails."""


class CLILoader(ContextLoader):
    """Loader for CLI contexts."""

    async def load(
        self,
        context: Context,
        processor_registry: ProcessorRegistry,
    ) -> LoadedContext:
        """Load context by executing CLI command.

        Args:
            context: Configuration for CLI-based context
            processor_registry: Registry containing available processors

        Returns:
            LoadedContext: Command output with metadata

        Raises:
            CLILoadError: If command execution fails or context type is invalid
        """
        if context.type != "cli":
            msg = f"Expected cli context, got {context.type}"
            raise CLILoadError(msg)

        # Now we know it's a CLIContext
        cli_context: CLIContext = context  # type: ignore

        try:
            # Prepare command and determine execution mode
            if cli_context.shell:
                if isinstance(cli_context.command, list):
                    cmd = " ".join(cli_context.command)
                else:
                    cmd = cli_context.command
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cli_context.cwd,
                )
            else:
                if isinstance(cli_context.command, str):
                    cmds = shlex.split(cli_context.command)
                else:
                    cmds = cli_context.command
                process = await asyncio.create_subprocess_exec(
                    *cmds,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cli_context.cwd,
                )

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=cli_context.timeout,
                )
            except TimeoutError as exc:
                process.kill()
                msg = (
                    f"Command timed out after {cli_context.timeout}s: "
                    f"{cli_context.command}"
                )
                raise CLILoadError(msg) from exc

            if process.returncode != 0:
                msg = (
                    f"Command failed with code {process.returncode}: "
                    f"{cli_context.command}\nError: {stderr.decode()}"
                )
                raise CLILoadError(msg)  # noqa: TRY301

            content = stdout.decode()

            # Process content through processors
            if cli_context.processors:
                content = await process_context(
                    content,
                    cli_context.processors,
                    processor_registry,
                )

            return LoadedContext(
                content=content,
                metadata={
                    "command": cli_context.command,
                    "shell": cli_context.shell,
                    "return_code": process.returncode,
                    "cwd": cli_context.cwd,
                },
            )

        except Exception as exc:
            if not isinstance(exc, CLILoadError):
                msg = f"Failed to execute command: {cli_context.command}"
                raise CLILoadError(msg) from exc
            raise


if __name__ == "__main__":
    from llmling.config import CLIContext

    async def main() -> None:
        # Simple example of CLI loader
        context = CLIContext(
            type="cli",
            command="git --help",
            description="Test command",
            shell=False,
        )

        loader = CLILoader()
        registry = ProcessorRegistry()

        try:
            result = await loader.load(context, registry)
            print(f"{result.content.strip()=}")
        except CLILoadError as exc:
            print(f"Error: {exc}")

    asyncio.run(main())
