from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from llmling.config.models import CLIContext
from llmling.context.base import ContextLoader
from llmling.context.models import LoadedContext
from llmling.core import exceptions
from llmling.core.log import get_logger


if TYPE_CHECKING:
    from llmling.config.models import Context
    from llmling.processors.registry import ProcessorRegistry


logger = get_logger(__name__)


class CLIContextLoader(ContextLoader):
    """Loads context from CLI command execution."""

    async def load(
        self,
        context: Context,
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
        if not isinstance(context, CLIContext):
            msg = f"Expected CLIContext, got {type(context).__name__}"
            raise exceptions.LoaderError(msg)

        try:
            cmd = (
                context.command
                if isinstance(context.command, str)
                else " ".join(context.command)
            )

            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=context.shell,
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
                raise exceptions.LoaderError(msg)

            content = stdout.decode()

            if context.processors:
                processed = await processor_registry.process(
                    content,
                    context.processors,
                )
                content = processed.content

            return LoadedContext(
                content=content,
                source_type="cli",
                metadata={
                    "type": "cli",
                    "command": context.command,
                    "exit_code": proc.returncode,
                    "size": len(content),
                },
            )

        except TimeoutError as exc:
            msg = f"Command timed out after {context.timeout} seconds"
            raise exceptions.LoaderError(msg) from exc
        except Exception as exc:
            msg = "CLI command execution failed"
            raise exceptions.LoaderError(msg) from exc
