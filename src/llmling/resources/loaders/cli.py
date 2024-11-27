"""CLI command context loader."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, ClassVar

from llmling.config.models import CLIResource
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.resources.base import ResourceLoader, create_loaded_resource


if TYPE_CHECKING:
    from llmling.processors.registry import ProcessorRegistry
    from llmling.resources.models import LoadedResource


logger = get_logger(__name__)


class CLIResourceLoader(ResourceLoader[CLIResource]):
    """Loads context from CLI command execution."""

    context_class = CLIResource
    uri_scheme = "cli"
    supported_mime_types: ClassVar[list[str]] = ["text/plain"]

    async def _load_impl(
        self,
        resource: CLIResource,
        name: str,
        processor_registry: ProcessorRegistry | None,
    ) -> LoadedResource:
        """Execute command and load output."""
        try:
            cmd = (
                resource.command
                if isinstance(resource.command, str)
                else " ".join(resource.command)
            )

            if resource.shell:
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=resource.cwd,
                )
            else:
                cmd_parts = cmd.split() if isinstance(cmd, str) else list(cmd)
                proc = await asyncio.create_subprocess_exec(
                    *cmd_parts,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=resource.cwd,
                )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=resource.timeout,
            )

            if proc.returncode != 0:
                error = stderr.decode().strip()
                msg = f"Command failed with code {proc.returncode}: {error}"
                raise exceptions.LoaderError(msg)  # noqa: TRY301

            content = stdout.decode()

            if processor_registry and (procs := resource.processors):
                processed = await processor_registry.process(content, procs)
                content = processed.content

            return create_loaded_resource(
                content=content,
                source_type="cli",
                uri=self.create_uri(name=name),
                mime_type=self.supported_mime_types[0],
                name=resource.description or f"CLI Output: {cmd}",
                description=resource.description,
                additional_metadata={
                    "command": cmd,
                    "exit_code": proc.returncode,
                },
            )
        except Exception as exc:
            msg = "CLI command execution failed"
            raise exceptions.LoaderError(msg) from exc
