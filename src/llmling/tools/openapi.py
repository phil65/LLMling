"""OpenAPI toolset implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

from llmling.core.log import get_logger
from llmling.tools.toolsets import ToolSet


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


logger = get_logger(__name__)


class OpenAPITools(ToolSet):
    """Tool collection for OpenAPI endpoints."""

    def __init__(
        self,
        spec: str,
        base_url: str = "",
        headers: dict[str, str] | None = None,
    ):
        import httpx

        self.spec_url = spec
        self.base_url = base_url
        self.headers = headers or {}
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=self.headers)
        self._spec: dict[str, Any] = {}
        self._operations: dict[str, dict[str, Any]] = {}
        self._factory: Any = None

    def _ensure_loaded(self):
        """Ensure spec is loaded."""
        if not self._spec:
            self._load_spec()

    def _load_spec(self):
        """Load and parse OpenAPI specification."""
        from schemez.openapi import OpenAPICallableFactory, load_openapi_spec, parse_operations

        try:
            self._spec = load_openapi_spec(self.spec_url)
        except Exception as exc:
            msg = f"Failed to load OpenAPI spec from {self.spec_url}"
            raise ValueError(msg) from exc

        # Get server URL if not overridden
        if not self.base_url and "servers" in self._spec:
            self.base_url = self._spec["servers"][0]["url"]

        # Parse operations
        paths = self._spec.get("paths", {})
        self._operations = parse_operations(paths)

        # Create factory for generating callables
        schemas = self._spec.get("components", {}).get("schemas", {})
        self._factory = OpenAPICallableFactory(schemas, self._request_handler)

    async def _request_handler(
        self,
        method: str,
        path: str,
        params: dict[str, Any],
        body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Handle HTTP requests for operations."""
        if not path.startswith("http"):
            path = urljoin(self.base_url.rstrip("/") + "/", path.lstrip("/"))

        response = await self._client.request(
            method=method,
            url=path,
            params=params,
            json=body,
        )
        response.raise_for_status()
        return response.json()

    def get_tools(self) -> list[Callable[..., Awaitable[dict[str, Any]]]]:
        """Get all API operations as tools."""
        self._ensure_loaded()
        return [
            self._factory.create_callable(op_id, config)
            for op_id, config in self._operations.items()
        ]


if __name__ == "__main__":

    async def main():
        url = "https://bird.ecb.europa.eu/documentation/api/v2/bird/bird-API-V2-documentation-Swagger-OpenAPI.yml"
        oapi = OpenAPITools(url)
        tools = oapi.get_tools()
        t = tools[0]
        result = await t(codes="ANCRDT_INSTRMNT_C")
        print(result)

    import asyncio

    asyncio.run(main())
