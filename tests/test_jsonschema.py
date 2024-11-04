from __future__ import annotations

from collections.abc import Sequence
import datetime as dt
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Literal

from pydantic import BaseModel
import pytest

from llmling.jsonschema import python_type_to_json_schema


# Constants for test values and expected schemas
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
DECIMAL_MULTIPLE = 0.01

BASIC_TYPES_MAPPING = {
    int: {"type": "integer"},
    str: {"type": "string"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    type(None): {"type": "null"},
}

FORMAT_TYPES_MAPPING = {
    dt.datetime: {"type": "string", "format": "date-time"},
    dt.date: {"type": "string", "format": "date"},
    Decimal: {"type": "number", "format": "decimal"},
}


# Test Fixtures
@pytest.fixture
def sample_enum() -> type[Enum]:
    class UserType(Enum):
        ADMIN = auto()
        USER = auto()
        GUEST = auto()

    return UserType


@pytest.fixture
def sample_pydantic_model() -> type[BaseModel]:
    class User(BaseModel):
        id: int
        name: str
        age: int | None
        created_at: dt.datetime

    return User


# Basic Type Tests
@pytest.mark.parametrize(
    "py_type,expected_schema",
    BASIC_TYPES_MAPPING.items(),
    ids=lambda x: str(x),
)
def test_basic_types(py_type: type, expected_schema: dict[str, Any]) -> None:
    schema = python_type_to_json_schema(py_type)
    assert schema == expected_schema


# Format Type Tests
@pytest.mark.parametrize(
    "py_type,expected_schema",
    FORMAT_TYPES_MAPPING.items(),
    ids=lambda x: str(x),
)
def test_format_types(py_type: type, expected_schema: dict[str, Any]) -> None:
    schema = python_type_to_json_schema(py_type)
    assert schema == expected_schema


# Enum Tests
def test_enum_schema(sample_enum: type[Enum]) -> None:
    schema = python_type_to_json_schema(sample_enum)
    assert schema["type"] == "string"
    assert set(schema["enum"]) == {"ADMIN", "USER", "GUEST"}


# Pydantic Model Tests
def test_pydantic_model_schema(sample_pydantic_model: type[BaseModel]) -> None:
    schema = python_type_to_json_schema(sample_pydantic_model)
    assert schema["type"] == "object"
    assert "properties" in schema
    assert set(schema["properties"]) == {"id", "name", "age", "created_at"}


# Container Type Tests
def test_list_type() -> None:
    schema = python_type_to_json_schema(list[str])
    assert schema == {"type": "array", "items": {"type": "string"}}


def test_dict_type() -> None:
    schema = python_type_to_json_schema(dict[str, int])
    assert schema == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }


# Union Type Tests
def test_union_type() -> None:
    schema = python_type_to_json_schema(int | str)
    assert schema == {"anyOf": [{"type": "integer"}, {"type": "string"}]}


def test_optional_type() -> None:
    schema = python_type_to_json_schema(str | None)
    assert schema == {"type": ["string", "null"]}


# Literal Type Tests
def test_literal_type() -> None:
    schema = python_type_to_json_schema(Literal["red", "blue", "green"])
    assert schema == {"enum": ["red", "blue", "green"]}


# Metadata Tests
def test_schema_with_metadata() -> None:
    schema = python_type_to_json_schema(
        int,
        title="Age",
        description="User's age",
        nullable=True,
    )
    assert schema == {
        "type": ["integer", "null"],
        "title": "Age",
        "description": "User's age",
    }


def test_schema_with_required_properties() -> None:
    class User(BaseModel):
        name: str
        age: int

    schema = python_type_to_json_schema(User)
    assert schema["required"] == ["name", "age"]


# Edge Cases and Error Tests
def test_sequence_handling() -> None:
    # Valid sequence
    schema = python_type_to_json_schema(Sequence[int])
    assert schema == {
        "type": "array",
        "items": {"type": "integer"},
    }

    # Undefined sequence
    with pytest.raises(
        ValueError, match="Cannot convert Python type Sequence to JSON Schema"
    ):
        python_type_to_json_schema(Sequence)


def test_unsupported_type() -> None:
    class CustomClass:
        pass

    with pytest.raises(ValueError, match="Cannot convert Python type"):
        python_type_to_json_schema(CustomClass)


def test_complex_nested_structure() -> None:
    class NestedModel(BaseModel):
        items: list[dict[str, int | None]]
        metadata: dict[str, Any]
        tags: set[str]

    schema = NestedModel.model_json_schema()
    # schema = python_type_to_json_schema(NestedModel)
    assert schema["type"] == "object"
    assert "properties" in schema
    assert set(schema["properties"]) == {"items", "metadata", "tags"}


# def test_recursive_type() -> None:
#     class RecursiveModel(BaseModel):
#         id: int
#         parent: RecursiveModel | None

#     schema = python_type_to_json_schema(RecursiveModel)
#     assert schema["type"] == "object"
#     assert "properties" in schema
#     assert set(schema["properties"]) == {"id", "parent"}


# Real-world complex example test
def test_complete_model() -> None:
    class UserStatus(Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"

    class Address(BaseModel):
        street: str
        city: str
        country: str
        postal_code: str | None

    class User(BaseModel):
        id: int
        email: str
        name: str | None
        status: UserStatus
        addresses: list[Address]
        created_at: dt.datetime
        balance: Decimal
        metadata: dict[str, Any]

    schema = python_type_to_json_schema(User)
    assert schema["type"] == "object"
    assert "properties" in schema
    required_fields = {
        "id",
        "email",
        "status",
        "addresses",
        "created_at",
        "balance",
        "metadata",
    }
    assert all(field in schema["properties"] for field in required_fields)
