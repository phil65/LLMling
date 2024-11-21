"""CLI command context loader."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import logfire

from llmling.config.models import CLIContext
from llmling.context.base import ContextLoader
from llmling.context.models import LoadedContext
from llmling.core import exceptions
from llmling.core.log import get_logger


if TYPE_CHECKING:
    from llmling.processors.registry import ProcessorRegistry


logger = get_logger(__name__)


class CLIContextLoader(ContextLoader[CLIContext]):
    """Loads context from CLI command execution."""

    context_class = CLIContext

    @logfire.instrument("Executing CLI command {context.command}")
    async def load(
        self,
        context: CLIContext,
        processor_registry: ProcessorRegistry,
    ) -> LoadedContext:
        """Load content from CLI command execution.

        Args:
            context: CLI context configuration
            processor_registry: Registry of available processors

        Returns:
            Loaded and processed context

        Raises:
            LoaderError: If command execution fails or context type is invalid
        """
        try:
            cmd = (
                context.command
                if isinstance(context.command, str)
                else " ".join(context.command)
            )

            if context.shell:
                # Use create_subprocess_shell when shell=True
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=context.cwd,
                )
            else:
                # Use create_subprocess_exec when shell=False
                if isinstance(context.command, str):
                    cmd_parts = cmd.split()
                else:
                    cmd_parts = list(context.command)

                proc = await asyncio.create_subprocess_exec(
                    *cmd_parts,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=context.cwd,
                )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=context.timeout,
            )

            if proc.returncode != 0:
                msg = (
                    f"Command failed with code {proc.returncode}: "
                    f"{stderr.decode().strip()}"
                )
                raise exceptions.LoaderError(msg)  # noqa: TRY301

            content = stdout.decode()

            if procs := context.processors:
                processed = await processor_registry.process(content, procs)
                content = processed.content
            meta = {
                "type": "cli",
                "command": context.command,
                "exit_code": proc.returncode,
                "size": len(content),
            }
            return LoadedContext(content=content, source_type="cli", metadata=meta)

        except TimeoutError as exc:
            msg = f"Command timed out after {context.timeout} seconds"
            raise exceptions.LoaderError(msg) from exc
        except Exception as exc:
            msg = "CLI command execution failed"
            raise exceptions.LoaderError(msg) from exc
