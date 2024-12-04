"""OpenAPI toolset implementation."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, TypeVar, Union
from uuid import UUID

import httpx
import upath
import yaml

from llmling.core.log import get_logger
from llmling.tools.toolsets import ToolSet


logger = get_logger(__name__)
T = TypeVar("T")

# Map OpenAPI formats to Python types
FORMAT_MAP = {
    "date": date,
    "date-time": datetime,
    "uuid": UUID,
    "email": str,
    "uri": str,
    "hostname": str,
    "ipv4": str,
    "ipv6": str,
    "byte": bytes,
    "binary": bytes,
    "password": str,
}


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
        self._schemas = self._spec.get("components", {}).get("schemas", {})
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

                # Collect all parameters (path, query, header)
                parameters = operation.get("parameters", [])
                if (
                    (request_body := operation.get("requestBody"))
                    and (content := request_body.get("content", {}))
                    and (json_schema := content.get("application/json", {}).get("schema"))
                    and (properties := json_schema.get("properties", {}))
                ):
                    # Convert request body to parameters
                    for name, schema in properties.items():
                        parameters.append({
                            "name": name,
                            "in": "body",
                            "required": name in json_schema.get("required", []),
                            "schema": schema,
                            "description": schema.get("description", ""),
                        })

                operations[op_id] = {
                    "method": method,
                    "path": path,
                    "description": operation.get("description", ""),
                    "parameters": parameters,
                    "responses": operation.get("responses", {}),
                }

        return operations

    def _resolve_schema_ref(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Resolve schema reference."""
        if ref := schema.get("$ref"):
            # Extract schema name from #/components/schemas/Name
            name = ref.split("/")[-1]
            return self._schemas[name]
        return schema

    def _get_type_for_schema(self, schema: dict[str, Any]) -> type | Any:  # noqa: PLR0911
        """Convert OpenAPI schema to Python type."""
        schema = self._resolve_schema_ref(schema)

        match schema.get("type"):
            case "string":
                if enum := schema.get("enum"):
                    return Literal[tuple(enum)]  # type: ignore
                if fmt := schema.get("format"):
                    return FORMAT_MAP.get(fmt, str)
                return str

            case "integer":
                return int

            case "number":
                return float

            case "boolean":
                return bool

            case "array":
                item_type = self._get_type_for_schema(schema["items"])
                return list[item_type]  # type: ignore

            case "object":
                if additional_props := schema.get("additionalProperties"):
                    # Dictionary with specified value type
                    value_type = self._get_type_for_schema(additional_props)
                    # Create type alias for the dict type

                    type DictType = dict[str, value_type]  # type: ignore
                    return DictType
                if _properties := schema.get("properties"):
                    # Convert to dict with specific types
                    return dict[str, Any]
                return dict[str, Any]

            case "null":
                return type(None)

            case None if "oneOf" in schema:
                types = [self._get_type_for_schema(s) for s in schema["oneOf"]]
                return Union[tuple(types)]  # type: ignore  # noqa: UP007

            case None if "anyOf" in schema:
                types = [self._get_type_for_schema(s) for s in schema["anyOf"]]
                return Union[tuple(types)]  # type: ignore  # noqa: UP007

            case None if "allOf" in schema:
                # For allOf, we'd need to merge schemas - using dict for now
                return dict[str, Any]

            case _:
                from typing import Any as AnyType

                return AnyType

    def _create_operation_method(
        self,
        op_id: str,
        config: dict[str, Any],
    ) -> Any:
        """Create a method for an operation with proper type hints."""
        # Create parameter annotations
        annotations: dict[str, Any] = {}
        required_params: set[str] = set()
        param_defaults: dict[str, Any] = {}

        for param in config["parameters"]:
            name = param["name"]
            schema = param.get("schema", {})

            # Get type
            param_type = self._get_type_for_schema(schema)
            annotations[name] = (
                param_type | None if not param.get("required") else param_type
            )

            # Track required params
            if param.get("required"):
                required_params.add(name)

            # Get default value if any
            if "default" in schema:
                param_defaults[name] = schema["default"]

        async def operation_method(**kwargs: Any) -> dict[str, Any]:
            """Dynamic method for API operation."""
            # Validate required parameters
            missing = required_params - set(kwargs)
            if missing:
                msg = f"Missing required parameters: {', '.join(missing)}"
                raise ValueError(msg)

            path = config["path"]
            request_params = {}
            request_body = {}

            # Process parameters based on their location
            for param in config["parameters"]:
                name = param["name"]
                if name not in kwargs and name in param_defaults:
                    kwargs[name] = param_defaults[name]

                if name in kwargs:
                    match param["in"]:
                        case "path":
                            path = path.replace(f"{{{name}}}", str(kwargs[name]))
                        case "query":
                            request_params[name] = kwargs[name]
                        case "body":
                            request_body[name] = kwargs[name]

            # Send request
            response = await self._client.request(
                method=config["method"],
                url=path,
                params=request_params,
                json=request_body if request_body else None,
            )
            response.raise_for_status()
            return response.json()

        # Set method metadata
        operation_method.__name__ = op_id
        operation_method.__doc__ = self._create_docstring(config)
        operation_method.__annotations__ = {**annotations, "return": dict[str, Any]}

        return operation_method

    def _create_docstring(self, config: dict[str, Any]) -> str:
        """Create detailed docstring from operation info."""
        lines = []
        if description := config["description"]:
            lines.append(description)
            lines.append("")

        # Add parameter descriptions
        if config["parameters"]:
            lines.append("Args:")
            for param in config["parameters"]:
                schema = param.get("schema", {})
                desc = param.get(
                    "description", schema.get("description", "No description")
                )
                required = " (required)" if param.get("required") else ""
                type_str = self._get_type_description(schema)
                lines.append(f"    {param['name']}: {desc}{required} ({type_str})")

        # Add response info
        if responses := config["responses"]:
            lines.append("")
            lines.append("Returns:")
            for code, response in responses.items():
                if code.startswith("2"):  # Success responses
                    desc = response.get("description", "")
                    lines.append(f"    {desc}")

        return "\n".join(lines)

    def _get_type_description(self, schema: dict[str, Any]) -> str:  # noqa: PLR0911
        """Get human-readable type description."""
        schema = self._resolve_schema_ref(schema)

        match schema.get("type"):
            case "string":
                if enum := schema.get("enum"):
                    return f"one of: {', '.join(repr(e) for e in enum)}"
                if fmt := schema.get("format"):
                    return f"string ({fmt})"
                return "string"

            case "array":
                item_type = self._get_type_description(schema["items"])
                return f"array of {item_type}"

            case "object":
                if properties := schema.get("properties"):
                    prop_types = [
                        f"{k}: {self._get_type_description(v)}"
                        for k, v in properties.items()
                    ]
                    return f"object with {', '.join(prop_types)}"
                return "object"

            case t:
                return str(t)

    def get_tools(self) -> list[Any]:
        """Get all API operations as tools."""
        return [
            self._create_operation_method(op_id, config)
            for op_id, config in self._operations.items()
        ]
