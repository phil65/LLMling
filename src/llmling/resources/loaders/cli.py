"""CLI command context loader."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import logfire

from llmling.config.models import CLIResource
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.resources.base import ResourceLoader, create_loaded_resource
from llmling.resources.models import LoadedResource


if TYPE_CHECKING:
    from llmling.processors.registry import ProcessorRegistry


logger = get_logger(__name__)


class CLIResourceLoader(ResourceLoader[CLIResource]):
    """Loads context from CLI command execution."""

    context_class = CLIResource
    uri_scheme = "cli"
    supported_mime_types = ["text/plain"]

    @logfire.instrument("Executing CLI command {context.command}")
    async def load(
        self,
        context: CLIResource,
        processor_registry: ProcessorRegistry,
    ) -> LoadedResource:
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

            return create_loaded_resource(
                content=content,
                source_type="cli",
                uri=self.create_uri(name=cmd.replace(" ", "-")),
                name=f"CLI Output: {cmd}",
                description=context.description,
                additional_metadata={"command": cmd, "exit_code": proc.returncode},
            )
        except Exception as exc:
            msg = "CLI command execution failed"
            raise exceptions.LoaderError(msg) from exc
