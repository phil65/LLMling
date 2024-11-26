"""Prompt-related protocol handlers."""

from __future__ import annotations

from mcp.types import (
    GetPromptResult,
    ListPromptsResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    ServerResult,
    TextContent,
)

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.server.handlers.base import HandlerBase


logger = get_logger(__name__)


class PromptHandlers(HandlerBase):
    """Prompt protocol handlers."""

    def register(self) -> None:
        """Register prompt handlers."""

        @self.server.server.list_prompts()
        async def handle_list_prompts() -> ServerResult:
            """List available prompts."""
            try:
                prompts = [
                    Prompt(
                        name=prompt.name,
                        description=prompt.description,
                        arguments=[
                            PromptArgument(
                                name=arg.name,
                                description=arg.description,
                                required=arg.required,
                            )
                            for arg in prompt.arguments
                        ],
                    )
                    for prompt in self.server.prompt_registry.values()
                ]
                return ServerResult(root=ListPromptsResult(prompts=prompts))
            except Exception as exc:
                logger.exception("Failed to list prompts")
                msg = f"Failed to list prompts: {exc}"
                raise exceptions.ProcessorError(msg) from exc

        @self.server.server.get_prompt()
        async def handle_get_prompt(
            name: str, arguments: dict[str, str] | None = None
        ) -> ServerResult:
            """Get a specific prompt."""
            try:
                # Render prompt
                result = await self.server.prompt_registry.render(name, arguments or {})

                # Convert to MCP messages
                messages = [
                    PromptMessage(
                        role=msg.role, content=TextContent(type="text", text=msg.content)
                    )
                    for msg in result.messages
                ]

                return ServerResult(
                    root=GetPromptResult(
                        description=f"Prompt: {name}",
                        messages=messages,
                    )
                )
            except Exception as exc:
                logger.exception("Failed to get prompt %s", name)
                msg = f"Failed to get prompt: {exc}"
                raise exceptions.ProcessorError(msg) from exc
