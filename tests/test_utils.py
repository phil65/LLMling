from __future__ import annotations

import datetime
import json
import os
from typing import TYPE_CHECKING, Any

import pytest

from llmling.utils.importing import import_callable


if TYPE_CHECKING:
    from collections.abc import Callable


class ExampleClass:
    """Test class for method imports."""

    @classmethod
    def class_method(cls) -> str:
        return "class_method"

    @staticmethod
    def static_method() -> str:
        return "static_method"

    def instance_method(self) -> str:
        return "instance_method"


@pytest.mark.parametrize(
    ("import_path", "expected_callable"),
    [
        # Builtins
        ("builtins.len", len),
        ("builtins.isinstance", isinstance),
        # Stdlib functions
        ("os.path.join", os.path.join),
        ("json.dumps", json.dumps),
        ("getpass.getpass", __import__("getpass").getpass),
        # Class methods
        ("datetime.datetime.now", datetime.datetime.now),
        # Static methods
        ("pathlib.Path.home", __import__("pathlib").Path.home),
        # Local class methods
        (
            f"{ExampleClass.__module__}.ExampleClass.class_method",
            ExampleClass.class_method,
        ),
        (
            f"{ExampleClass.__module__}.ExampleClass.static_method",
            ExampleClass.static_method,
        ),
        # Instance methods
        (
            f"{ExampleClass.__module__}.ExampleClass.instance_method",
            ExampleClass.instance_method,
        ),
    ],
)
def test_import_callable(import_path: str, expected_callable: Callable[..., Any]) -> None:
    """Test importing various types of callables.

    Args:
        import_path: Import path to test
        expected_callable: The callable we expect to get
    """
    imported = import_callable(import_path)
    assert imported == expected_callable


@pytest.mark.parametrize(
    "invalid_path",
    [
        "nonexistent.module.func",  # non-existent module
        "sys.path",  # exists but not callable
        "",  # empty string
        "os.path..join",  # invalid dots
        ":invalid:colons:",  # invalid colons
    ],
)
def test_import_callable_invalid(invalid_path: str) -> None:
    """Test invalid import paths raise ValueError."""
    with pytest.raises(Exception):  # noqa: B017, PT011
        import_callable(invalid_path)
