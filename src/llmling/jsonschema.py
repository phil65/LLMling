"""Convert Python types and objects to JSON Schema specifications.

This module provides functionality to convert Python types, including basic types,
enums, Pydantic models, and typing annotations to their equivalent JSON Schema
representations.

Example:
    ```python
    from datetime import datetime
    from pydantic import BaseModel
    from enum import Enum

    class UserType(Enum):
        ADMIN = "admin"
        USER = "user"

    class User(BaseModel):
        id: int
        name: str
        created_at: datetime
        type: UserType

    schema = python_type_to_json_schema(User)
    ```
"""

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
import datetime as dt
import decimal
import enum
import inspect
import types
import typing
from typing import Any, get_args, get_origin
import uuid

import pydantic


type JsonSchema = dict[str, Any]

JSON_SCHEMA_TYPE_MAPPING: dict[type, str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    dict: "object",
    type(None): "null",
    decimal.Decimal: "number",
    dt.datetime: "string",
    dt.date: "string",
    uuid.UUID: "string",
}

JSON_SCHEMA_FORMAT_MAPPING: dict[type, str] = {
    dt.datetime: "date-time",
    dt.date: "date",
    uuid.UUID: "uuid",
    decimal.Decimal: "decimal",
}

JSON_SCHEMA_CONSTRAINTS: dict[type, dict[str, Any]] = {
    int: {"minimum": -(2**31), "maximum": 2**31 - 1},  # default to int32
    decimal.Decimal: {"multipleOf": 0.01},  # default to 2 decimal places
}


def python_type_to_json_schema(
    py_type: Any,
    *,
    title: str | None = None,
    description: str | None = None,
    nullable: bool = False,
    required: list[str] | None = None,
) -> JsonSchema:
    """Convert a Python type or object to its JSON Schema representation.

    Converts various Python types including basic types, enums, Pydantic models,
    and typing annotations to their equivalent JSON Schema representation.

    Args:
        py_type: Python type or instance to convert. Can be a basic type (int, str),
            typing annotation (list[str]), Pydantic model, or enum.
        title: Optional title for the schema. Useful for documentation.
        description: Optional description explaining the schema's purpose.
        nullable: If True, adds null as an acceptable type.
        required: List of field names that are required (only for object schemas).

    Returns:
        A dictionary containing the JSON Schema representation.

    Raises:
        ValueError: If the input type cannot be mapped to a JSON Schema type.

    Examples:
        Basic type with metadata:
        ```python
        schema = python_type_to_json_schema(
            int,
            title="Age",
            description="User's age in years",
            nullable=True
        )
        ```

        Complex Pydantic model:
        ```python
        class User(BaseModel):
            id: int
            name: str
            tags: list[str]

        schema = python_type_to_json_schema(User)
        ```

        Generic container types:
        ```python
        schema = python_type_to_json_schema(list[str])
        schema = python_type_to_json_schema(dict[str, int])
        ```
    """
    schema = _build_base_schema(py_type)

    # Add optional metadata
    if title:
        schema["title"] = title
    if description:
        schema["description"] = description
    if nullable:
        schema["type"] = [schema["type"], "null"]
    if required and schema.get("type") == "object":
        schema["required"] = required

    return schema


def _build_base_schema(py_type: Any) -> JsonSchema:
    """Build the base JSON Schema representation without additional metadata.

    This is the core conversion function that handles different Python types and
    returns their base JSON Schema representation.

    Args:
        py_type: The Python type or object to convert

    Returns:
        The base JSON Schema representation without additional metadata

    Raises:
        ValueError: If the type cannot be converted to JSON Schema
    """
    # Handle direct type mappings
    if isinstance(py_type, type) and py_type in JSON_SCHEMA_TYPE_MAPPING:
        schema = {"type": JSON_SCHEMA_TYPE_MAPPING[py_type]}
        # Add format if applicable
        if format_type := JSON_SCHEMA_FORMAT_MAPPING.get(py_type):
            schema["format"] = format_type

        # Add constraints if applicable
        # if constraints := JSON_SCHEMA_CONSTRAINTS.get(py_type):
        #     schema.update(constraints)

        return schema

    # Handle typing annotations
    if origin := get_origin(py_type):
        return _handle_typing_annotation(py_type, origin)
    # Handle enums
    if isinstance(py_type, type) and issubclass(py_type, enum.Enum):
        return _handle_enum(py_type)
    # Handle Pydantic models
    if isinstance(py_type, type) and issubclass(py_type, pydantic.BaseModel):
        return _handle_pydantic_model(py_type)

    msg = f"Cannot convert Python type {py_type.__name__} to JSON Schema"
    raise ValueError(msg)


def _handle_typing_annotation(py_type: Any, origin: type) -> JsonSchema:
    """Handle typing module annotations and convert to JSON Schema.

    Args:
        py_type: The typing annotation to convert
        origin: The base type of the annotation (from get_origin)

    Returns:
        The JSON Schema representation of the typing annotation

    Raises:
        ValueError: If the typing annotation is not supported
    """
    args = get_args(py_type)

    # Handle Literal types
    if origin is typing.Literal:
        return {"enum": list(args)}

    # Handle sequence-like types
    if (
        origin in (list, builtins.list)
        or isinstance(origin, type)
        and issubclass(origin, Sequence)
    ):
        return {
            "type": "array",
            "items": _build_base_schema(args[0]) if args else {"type": "object"},
        }

    # Handle dictionary types
    if origin in (dict, builtins.dict):
        return {
            "type": "object",
            "additionalProperties": _build_base_schema(args[1])
            if args
            else {"type": "object"},
        }

    # Handle Union types
    if origin in (typing.Union, types.UnionType):
        non_none_types = [
            _build_base_schema(arg) for arg in args if arg is not type(None)
        ]
        if len(non_none_types) == 1:
            schema = non_none_types[0]
            schema["type"] = [schema["type"], "null"]
            return schema
        return {"anyOf": non_none_types}

    msg = f"Unsupported typing annotation: {py_type}"
    raise ValueError(msg)


def _handle_enum(enum_type: type[enum.Enum]) -> JsonSchema:
    """Convert an Enum class to its JSON Schema representation.

    Converts Python enum classes to a JSON Schema enum, using the enum member
    names as the allowed values.

    Args:
        enum_type: The Enum class to convert

    Returns:
        A JSON Schema object with type "string" and enum values

    Example:
        ```python
        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        schema = _handle_enum(Color)
        # Returns: {
        #     "type": "string",
        #     "enum": ["RED", "BLUE"],
        #     "description": <enum docstring if present>
        # }
        ```
    """
    return {
        "type": "string",
        "enum": [e.name for e in enum_type],
        "description": inspect.getdoc(enum_type),
    }


def _handle_pydantic_model(model_type: type[pydantic.BaseModel]) -> JsonSchema:
    """Handle Pydantic models."""
    return model_type.model_json_schema()


def _handle_mapping(mapping: Mapping[str, Any]) -> JsonSchema:
    """Handle mapping types."""
    try:
        properties = {
            str(key): _build_base_schema(type(value)) for key, value in mapping.items()
        }
    except (TypeError, ValueError) as exc:
        msg = "Invalid dictionary structure for JSON Schema conversion"
        raise ValueError(msg) from exc
    else:
        return {"type": "object", "properties": properties}


if __name__ == "__main__":
    from datetime import datetime
    from decimal import Decimal
    from enum import Enum

    from pydantic import BaseModel

    # Basic types
    schema = python_type_to_json_schema(
        int, title="Age", description="User age", nullable=True
    )
    print(schema)

    # Enums
    class UserType(Enum):
        ADMIN = "admin"
        USER = "user"

    schema = python_type_to_json_schema(UserType)
    print(schema)

    # Pydantic models
    class User(BaseModel):
        id: int
        name: str
        created_at: datetime
        balance: Decimal
        type: UserType
        tags: list[str]
        metadata: dict[str, Any] | None

    schema = python_type_to_json_schema(User)
    print(schema)

    # Typing annotations
    schema = python_type_to_json_schema(list[str])
    print(schema)
    schema = python_type_to_json_schema(dict[str, int])
    print(schema)
    schema = python_type_to_json_schema(str | None)
    print(schema)

    # Complex structures
    data = {
        "users": [
            {"name": "John", "age": 30},
            {"name": "Jane", "age": 25},
        ]
    }
    schema = python_type_to_json_schema(data)
    print(schema)
