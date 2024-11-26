"""Resource-related protocol handlers."""

from __future__ import annotations

from mcp.types import (
    BlobResourceContents,
    EmptyResult,
    ListResourcesResult,
    ReadResourceResult,
    Resource,
    ServerResult,
    TextResourceContents,
)
from pydantic import AnyUrl

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.server.handlers.base import HandlerBase


logger = get_logger(__name__)


class ResourceHandlers(HandlerBase):
    """Resource protocol handlers."""

    def register(self) -> None:
        """Register resource handlers."""

        @self.server.server.list_resources()
        async def handle_list_resources() -> ServerResult:
            """List available resources."""
            try:
                resources = []
                # Convert contexts to MCP resources
                for name, context in self.server.config.contexts.items():
                    # Get loader for context type
                    loader_class = self.server.resource_registry[context.context_type]
                    loader = loader_class.create(context)
                    uri = loader.create_uri(name=name)

                    resources.append(
                        Resource(
                            uri=AnyUrl(uri),
                            name=name,
                            description=context.description,
                            mimeType=loader.supported_mime_types[0],
                        )
                    )

                return ServerResult(root=ListResourcesResult(resources=resources))
            except Exception as exc:
                logger.exception("Failed to list resources")
                msg = "Failed to list resources"
                raise exceptions.ResourceError(msg) from exc

        @self.server.server.read_resource()
        async def handle_read_resource(uri: AnyUrl) -> ServerResult:
            """Read a specific resource."""
            try:
                # Get loader for URI
                uri_str = str(uri)
                loader = self.server.resource_registry.find_loader_for_uri(uri_str)

                # Track progress
                await self.server.notify_progress(
                    0.0, 1.0, description=f"Loading resource {uri_str}"
                )

                # Load resource
                result = await loader.load(
                    context=loader.context,
                    processor_registry=self.server.processor_registry,
                )

                # Complete progress
                await self.server.notify_progress(
                    1.0, 1.0, description="Resource loaded successfully"
                )

                # Create appropriate content type
                if isinstance(result.content, str):
                    content = TextResourceContents(
                        uri=uri,
                        text=result.content,
                        mimeType=loader.supported_mime_types[0],
                    )
                else:
                    import base64

                    content = BlobResourceContents(
                        uri=uri,
                        blob=base64.b64encode(result.content).decode(),
                        mimeType=loader.supported_mime_types[0],
                    )

                return ServerResult(root=ReadResourceResult(contents=[content]))
            except Exception as exc:
                logger.exception("Failed to read resource %s", uri)
                msg = f"Failed to read resource: {exc}"
                raise exceptions.ResourceError(msg) from exc

        @self.server.server.subscribe_resource()
        async def handle_subscribe_resource(uri: AnyUrl) -> ServerResult:
            """Handle resource subscription."""
            try:
                # Store subscription
                uri_str = str(uri)
                if not hasattr(self.server, "_subscriptions"):
                    self.server._subscriptions = {}
                if uri_str not in self.server._subscriptions:
                    self.server._subscriptions[uri_str] = set()
                self.server._subscriptions[uri_str].add(uri)
                return ServerResult(root=EmptyResult())
            except Exception as exc:
                logger.exception("Failed to subscribe to resource %s", uri)
                msg = f"Failed to subscribe: {exc}"
                raise exceptions.ResourceError(msg) from exc

        @self.server.server.unsubscribe_resource()
        async def handle_unsubscribe_resource(uri: AnyUrl) -> ServerResult:
            """Handle resource unsubscription."""
            try:
                # Remove subscription
                uri_str = str(uri)
                if hasattr(self.server, "_subscriptions"):
                    if uri_str in self.server._subscriptions:
                        self.server._subscriptions[uri_str].discard(uri)
                        if not self.server._subscriptions[uri_str]:
                            del self.server._subscriptions[uri_str]
                return ServerResult(root=EmptyResult())
            except Exception as exc:
                logger.exception("Failed to unsubscribe from resource %s", uri)
                raise exceptions.ResourceError(f"Failed to unsubscribe: {exc}") from exc
