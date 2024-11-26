"""Completion-related protocol handlers."""

from __future__ import annotations

import glob
import os
from typing import TYPE_CHECKING

from mcp.types import (
    CompleteRequestParams,
    CompleteResult,
    Completion,
    ServerResult,
)

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.prompts.models import ArgumentType
from llmling.server.handlers.base import HandlerBase


if TYPE_CHECKING:
    from llmling.prompts.models import ExtendedPromptArgument

logger = get_logger(__name__)


class CompletionHandlers(HandlerBase):
    """Completion protocol handlers."""

    def register(self) -> None:
        """Register completion handlers."""

        @self.server.server.complete()
        async def handle_complete(params: CompleteRequestParams) -> ServerResult:
            """Handle completion requests."""
            try:
                match params.ref.type:
                    case "ref/prompt":
                        # Get prompt and find argument
                        prompt = self.server.prompt_registry.get(params.ref.name)
                        arg = next(
                            (
                                a
                                for a in prompt.arguments
                                if a.name == params.argument.name
                            ),
                            None,
                        )
                        if not arg:
                            msg = f"Argument not found: {params.argument.name}"
                            raise exceptions.ProcessorError(msg)

                        # Get completions based on argument type
                        values = await self._get_completions(
                            arg,
                            params.argument.value,
                        )

                    case "ref/resource":
                        # Handle resource URI completion
                        values = await self._complete_resource_uri(params.ref.uri)

                    case _:
                        msg = f"Unsupported reference type: {params.ref.type}"
                        raise exceptions.ProcessorError(msg)

                return ServerResult(
                    root=CompleteResult(
                        completion=Completion(
                            values=values[:100],  # Limit to 100 as per spec
                            total=len(values),
                            hasMore=len(values) > 100,
                        )
                    )
                )

            except Exception as exc:
                logger.exception("Completion failed")
                msg = f"Failed to get completions: {exc}"
                raise exceptions.ProcessorError(msg) from exc

    async def _get_completions(
        self,
        argument: ExtendedPromptArgument,
        current: str,
    ) -> list[str]:
        """Get completion values for an argument."""
        match argument.type:
            case ArgumentType.ENUM:
                return [v for v in argument.enum_values or [] if v.startswith(current)]

            case ArgumentType.FILE:
                return await self._complete_file_path(
                    current,
                    patterns=argument.file_patterns,
                )

            case ArgumentType.TOOL:
                allowed = set(argument.tool_names or ["*"])
                names = self.server.tool_registry.list_items()
                if "*" not in allowed:
                    names = [n for n in names if n in allowed]
                return [n for n in names if n.startswith(current)]

            case ArgumentType.RESOURCE:
                # Get resources from registry instead of handler
                allowed = set(argument.resource_types or ["*"])
                # Get all resources from our registry
                resources = self.server.resource_registry.list_items()
                uris = []
                for resource_type, loader in resources.items():
                    if "*" in allowed or resource_type in allowed:
                        uri = loader.get_uri_template()
                        if uri.startswith(current):
                            uris.append(uri)
                return uris

            case _:
                return []  # No completions for plain text

    async def _complete_file_path(
        self,
        current: str,
        patterns: list[str] | None = None,
    ) -> list[str]:
        """Get file path completions."""
        if not current:
            return []

        # Get base directory and pattern
        base_dir = os.path.dirname(current) or "."
        name_pattern = os.path.basename(current)

        # Collect matching files
        matches = []
        for pattern in patterns or ["*"]:
            # Combine current path with pattern
            full_pattern = os.path.join(base_dir, f"{name_pattern}*{pattern}")
            matches.extend(glob.glob(full_pattern))

        # Normalize paths
        return [os.path.normpath(p) for p in matches]

    async def _complete_resource_uri(self, uri_prefix: str) -> list[str]:
        """Complete resource URIs."""
        try:
            # Use resource registry directly instead of handler
            resources = []
            for resource_type, loader in self.server.resource_registry.items():
                uri = loader.get_uri_template()
                if uri.startswith(uri_prefix):
                    resources.append(uri)
            return resources
        except Exception as exc:
            logger.warning("Failed to complete resource URI: %s", exc)
            return []
