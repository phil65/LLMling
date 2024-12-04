"""CLI command context loader."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar

from llmling.config.models import CLIResource
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.resources.base import ResourceLoader, create_loaded_resource


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

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
    ) -> AsyncIterator[LoadedResource]:
        """Execute command and load output."""
        command = cmd if isinstance((cmd := resource.command), str) else " ".join(cmd)
        try:
            kwargs: dict[str, Any] = dict(
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=resource.cwd,
            )
            if resource.shell:
                proc = await asyncio.create_subprocess_shell(command, **kwargs)
            else:
                proc = await asyncio.create_subprocess_exec(*command.split(), **kwargs)
            coro = proc.communicate()
            stdout, stderr = await asyncio.wait_for(coro, timeout=resource.timeout)

            if proc.returncode != 0:
                error = stderr.decode().strip()
                msg = f"Command failed with code {proc.returncode}: {error}"
                raise exceptions.LoaderError(msg)  # noqa: TRY301

            content = stdout.decode()

            if processor_registry and (procs := resource.processors):
                processed = await processor_registry.process(content, procs)
                content = processed.content
            meta = {"command": command, "exit_code": proc.returncode}
            yield create_loaded_resource(
                content=content,
                source_type="cli",
                uri=self.create_uri(name=name),
                mime_type=self.supported_mime_types[0],
                name=resource.description or f"CLI Output: {command}",
                description=resource.description,
                additional_metadata=meta,
            )
        except Exception as exc:
            msg = f"CLI command execution failed: {command}"
            raise exceptions.LoaderError(msg) from exc
