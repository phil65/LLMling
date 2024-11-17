"""Tool loading and schema generation utilities."""

from __future__ import annotations

from importlib import import_module
import inspect
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable
    import types


class ToolError(Exception):
    """Base exception for tool-related errors."""


def load_tool(import_path: str) -> Callable[..., Any]:
    """Load a tool function or class from import path."""
    try:
        module_path, attr_name = import_path.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, attr_name)
    except (ImportError, AttributeError) as exc:
        msg = f"Failed to import tool from {import_path}"
        raise ToolError(msg) from exc


def generate_schema_from_function(func: Callable[..., Any]) -> dict[str, Any]:
    """Generate OpenAPI-compatible schema from function signature."""
    sig = inspect.signature(func)

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            required.append(name)

        param_type = param.annotation
        if param_type is inspect.Parameter.empty:
            param_type = Any

        properties[name] = {
            "type": _get_type_str(param_type),
            "description": _get_param_doc(func, name),
        }

        if param.default is not inspect.Parameter.empty:
            properties[name]["default"] = param.default

    return {"type": "object", "properties": properties, "required": required}


def generate_schema_from_class(cls: type) -> dict[str, dict[str, Any]]:
    """Generate schemas for all public methods in a class."""
    schemas = {}

    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if not name.startswith("_"):
            schemas[f"{cls.__name__}.{name}"] = generate_schema_from_function(method)

    return schemas


def generate_schema_from_module(
    module: types.ModuleType, include_functions: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Generate schemas for all functions in a module."""
    schemas = {}

    for name, func in inspect.getmembers(module, predicate=inspect.isfunction):
        if (
            include_functions is None or name in include_functions
        ) and not name.startswith("_"):
            schemas[name] = generate_schema_from_function(func)
    return schemas


def _get_type_str(typ: type) -> str:
    """Convert Python type to JSON schema type string."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    return type_map.get(typ, "string")


def _get_param_doc(func: Callable[..., Any], param_name: str) -> str:
    """Extract parameter description from function docstring."""
    if func.__doc__ is None:
        return ""

    docstring = inspect.cleandoc(func.__doc__)

    # Simple parsing - could be enhanced with docstring parsers
    for line in docstring.split("\n"):
        if f"{param_name}:" in line:
            return line.split(":", 1)[1].strip()
    return ""


if __name__ == "__main__":

    def example(x: int, y: str = "test") -> None:
        """Example function.

        Args:
            x: A number
            y: A string
        """

    schema = generate_schema_from_function(example)
    print(schema)
