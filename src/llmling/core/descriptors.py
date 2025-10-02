"""Descriptor utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable


class classproperty[ReturnT]:  # noqa: N801
    """Descriptor for class-level properties."""

    def __init__(self, method: Callable[[type], ReturnT] | None = None) -> None:
        self.method = method

    def __get__(self, instance: Any, cls: type | None = None) -> ReturnT:
        if cls is None:
            cls = type(instance)
        if self.method is None:
            msg = "No method defined"
            raise AttributeError(msg)
        return self.method(cls)

    def __set__(self, instance: Any, value: Any) -> None:
        msg = "Can't set class property"
        raise AttributeError(msg)
