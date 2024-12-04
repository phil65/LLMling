"""OpenAPI toolset implementation."""

from __future__ import annotations

from typing import Any

import httpx
import upath
import yaml

from llmling.core.log import get_logger
from llmling.tools.toolsets import ToolSet


logger = get_logger(__name__)


class OpenAPITools(ToolSet):
    """Tool collection for OpenAPI endpoints."""

    def __init__(
        self,
        spec: str,
        base_url: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize OpenAPI tools.

        Args:
            spec: URL or path to OpenAPI spec
            base_url: Optional base URL override
            headers: Optional headers for requests
        """
        self.spec_url = spec
        self.base_url = base_url
        self.headers = headers or {}
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
        )
        self._spec = self._load_spec()
        self._operations = self._parse_operations()

    def _load_spec(self) -> dict[str, Any]:
        """Load OpenAPI specification."""
        try:
            if self.spec_url.startswith(("http://", "https://")):
                response = httpx.get(self.spec_url)
                response.raise_for_status()
                content = response.text
            else:
                with upath.UPath(self.spec_url).open() as f:
                    content = f.read()

            return yaml.safe_load(content)
        except Exception as exc:
            msg = f"Failed to load OpenAPI spec from {self.spec_url}"
            raise ValueError(msg) from exc

    def _parse_operations(self) -> dict[str, dict[str, Any]]:
        """Parse OpenAPI spec into operation configurations."""
        operations = {}

        # Get server URL if not overridden
        if not self.base_url and "servers" in self._spec:
            self.base_url = self._spec["servers"][0]["url"]

        # Parse paths and operations
        for path, path_item in self._spec.get("paths", {}).items():
            for method, operation in path_item.items():
                if method not in {"get", "post", "put", "delete", "patch"}:
                    continue

                # Generate operation ID if not provided
                op_id = operation.get("operationId")
                if not op_id:
                    op_id = f"{method}_{path.replace('/', '_').strip('_')}"

                operations[op_id] = {
                    "method": method,
                    "path": path,
                    "description": operation.get("description", ""),
                    "parameters": operation.get("parameters", []),
                }

        return operations

    def _create_operation_method(
        self,
        op_id: str,
        config: dict[str, Any],
    ) -> Any:
        """Create a method for an operation."""

        async def operation_method(**kwargs: Any) -> Any:
            """Dynamic method for API operation."""
            path = config["path"]

            # Replace path parameters
            for param in config["parameters"]:
                if param["in"] == "path":
                    name = param["name"]
                    if name in kwargs:
                        path = path.replace(f"{{{name}}}", str(kwargs[name]))
                        kwargs.pop(name)

            # Send request
            response = await self._client.request(
                method=config["method"],
                url=path,
                params=kwargs if config["method"] == "get" else None,
                json=kwargs if config["method"] != "get" else None,
            )
            response.raise_for_status()
            return response.json()

        # Set method metadata
        operation_method.__name__ = op_id
        operation_method.__doc__ = config["description"]

        return operation_method

    def get_tools(self) -> list[Any]:
        """Get all API operations as tools."""
        return [
            self._create_operation_method(op_id, config)
            for op_id, config in self._operations.items()
        ]
