"""High-level interface for LLMling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from llmling.client import LLMLingClient
from llmling.config.loading import load_config
from llmling.config.models import TaskSettings, TaskTemplate, TextContext
from llmling.core import exceptions
from llmling.core.log import get_logger, setup_logging
from llmling.llm.base import Message


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    import os
    from types import TracebackType

    from llmling.core.typedefs import ProcessingStep


logger = get_logger(__name__)


class Chat:
    def __init__(
        self,
        client: LLMLingClient,
        provider: str,
        *,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize chat session."""
        self.client = client
        self.provider = provider
        self.messages: list[Message] = []

        # Create template name for this chat session
        self._template_name = f"chat_{provider}"

        # Ensure chat context exists
        if not client.config_manager:
            msg = "Client not properly initialized"
            raise exceptions.LLMLingError(msg)

        # Add chat context if needed
        context = TextContext(
            type="text",
            # Use format-style template for runtime context
            content="{message}",  # Use simple format string
            description="Dynamic chat context",
            processors=[],
        )
        client.config_manager.config.contexts["chat"] = context

        # Create chat template
        if self._template_name not in client.config_manager.config.task_templates:
            client.config_manager.config.task_templates[self._template_name] = (
                TaskTemplate(
                    provider=self.provider,
                    context="chat",
                    settings=TaskSettings(
                        temperature=0.7,
                        max_tokens=2048,
                    ),
                )
            )

        # Add system prompt if provided
        if system_prompt:
            self.messages.append(Message(role="system", content=system_prompt))

    def send(
        self,
        message: str,
        **kwargs: Any,
    ) -> str:
        """Send a message and get response."""
        # Add user message to history
        self.messages.append(Message(role="user", content=message))

        # Create context dictionary with just the message
        context = {
            "message": message,  # Match the format string in context
        }

        # Execute template
        result = self.client.execute_sync(
            self._template_name,
            context=context,
            **kwargs,
        )

        # Add assistant's response to history
        self.messages.append(Message(role="assistant", content=result.content))

        return result.content

    async def send_async(
        self,
        message: str,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | AsyncIterator[str]:
        """Send a message asynchronously."""
        # Add user message to history
        self.messages.append(Message(role="user", content=message))

        # Create context dictionary with messages and new message
        context = {
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                }
                for msg in self.messages
            ],
            "new_message": message,
        }

        if stream:
            # Stream responses
            response_content = []
            async for result in self.client.stream(
                self._template_name,
                context=context,
                **kwargs,
            ):
                response_content.append(result.content)
                yield result.content

            # Add complete assistant response after streaming
            self.messages.append(
                Message(role="assistant", content="".join(response_content))
            )
        else:
            # Single response
            result = await self.client.execute(
                self._template_name,
                context=context,
                **kwargs,
            )
            self.messages.append(Message(role="assistant", content=result.content))
            yield result.content

    def add_context(
        self,
        source: str | os.PathLike[str],
        *,
        context_type: str | None = None,
        processors: list[ProcessingStep] | None = None,
    ) -> None:
        """Add context to the conversation.

        Args:
            source: Path or URL to context
            context_type: Optional context type
            processors: Optional list of processors to apply
        """
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            context = loop.run_until_complete(
                self.client.manager.load_context(
                    str(source),
                    context_type=context_type,
                    processors=processors,
                )
            )
            if context.content:
                self.messages.append(Message(role="system", content=context.content))
            if context.content_items:
                for item in context.content_items:
                    self.messages.append(
                        Message(
                            role="system",
                            content=item.content,
                            content_items=[item],
                        )
                    )
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    async def add_context_async(
        self,
        source: str | os.PathLike[str],
        *,
        context_type: str | None = None,
        processors: list[ProcessingStep] | None = None,
    ) -> None:
        """Add context asynchronously.

        Args:
            source: Path or URL to context
            context_type: Optional context type
            processors: Optional list of processors to apply
        """
        context = await self.client.manager.load_context(
            str(source),
            context_type=context_type,
            processors=processors,
        )
        if context.content:
            self.messages.append(Message(role="system", content=context.content))
        if context.content_items:
            for item in context.content_items:
                self.messages.append(
                    Message(
                        role="system",
                        content=item.content,
                        content_items=[item],
                    )
                )

    def add_system_prompt(self, prompt: str) -> None:
        """Add system prompt to the conversation."""
        self.messages.append(Message(role="system", content=prompt))


class LLMling:
    """Simple interface for LLMling functionality."""

    def __init__(
        self,
        config_path: str | os.PathLike[str],
        *,
        log_level: str | None = None,
    ) -> None:
        """Initialize LLMling."""
        if log_level:
            setup_logging(level=log_level)

        self.config_path = config_path
        self._config = load_config(config_path)

        # Ensure required contexts exist
        context = TextContext(
            type="text",
            content="{{ new_message }}",  # Template variable for new message
            description="Dynamic chat context",
        )
        self._config.contexts["chat"] = context

        # Create client with updated config
        self._client = LLMLingClient(config_path)
        self._initialized = False

    def chat_with(
        self,
        provider: str,
        *,
        system_prompt: str | None = None,
    ) -> Chat:
        """Create a new chat session with a provider."""
        # Validate provider exists
        if provider not in self._config.llm_providers:
            msg = f"Provider {provider} not found"
            raise exceptions.LLMLingError(msg)

        return Chat(self.client, provider, system_prompt=system_prompt)

    async def startup(self) -> None:
        """Initialize components."""
        if self._initialized:
            return

        try:
            await self._client.startup()
            self._initialized = True
            logger.debug("LLMling initialized successfully")

        except Exception as exc:
            logger.exception("Initialization failed")
            await self.shutdown()
            msg = f"Failed to initialize LLMling: {exc}"
            raise exceptions.LLMLingError(msg) from exc

    @property
    def client(self) -> LLMLingClient:
        """Get the LLMling client, ensuring it's initialized."""
        if not self._initialized:
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.startup())
            finally:
                asyncio.set_event_loop(None)
                loop.close()

        return self._client

    @property
    def providers(self) -> list[str]:
        """Get available provider names."""
        return list(self._config.llm_providers)

    @property
    def provider_groups(self) -> list[str]:
        """Get available provider group names."""
        return list(self._config.provider_groups)

    @property
    def tools(self) -> list[str]:
        """Get available tool names."""
        return list(self._config.tools) if self._config.tools else []

    def run_tool(
        self,
        name: str,
        **params: Any,
    ) -> Any:
        """Execute a tool.

        Args:
            name: Tool name
            **params: Tool parameters

        Returns:
            Tool execution result
        """
        # Pass tool name and parameters as context variables
        result = self.client.execute_sync(
            "tool_execution",  # Template name
            context={  # Context variables
                "tool": name,
                "parameters": params,
            },
        )
        yield result.content

    async def run_tool_async(
        self,
        name: str,
        **params: Any,
    ) -> Any:
        """Execute a tool asynchronously.

        Args:
            name: Tool name
            **params: Tool parameters

        Returns:
            Tool execution result
        """
        result = await self.client.execute(
            "tool_execution",  # Template name
            context={  # Context variables
                "tool": name,
                "parameters": params,
            },
        )
        return result.content

    async def shutdown(self) -> None:
        """Clean up resources."""
        if not self._initialized:
            return

        try:
            if self._client:
                await self._client.shutdown()
        except Exception as exc:
            logger.exception("Error during shutdown")
            msg = f"Failed to shutdown LLMling: {exc}"
            raise exceptions.LLMLingError(msg) from exc
        finally:
            self._initialized = False
            logger.info("LLMling shut down successfully")

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        await self.startup()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        await self.shutdown()

    def __enter__(self) -> Self:
        """Context manager entry."""
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.startup())
            return self
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.shutdown())
        finally:
            asyncio.set_event_loop(None)
            loop.close()


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    with LLMling("src/llmling/resources/test.yml") as llm:
        chat = llm.chat_with("gpt-35-turbo")
        response = chat.send("Hello")
        print(response)
