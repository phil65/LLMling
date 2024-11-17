from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import logfire


if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def capture_spans() -> Generator[list[dict[str, Any]], None, None]:
    """Capture spans during a test."""
    spans: list[dict[str, Any]] = []

    def callback(span: dict[str, Any]) -> None:
        spans.append(span)

    token = logfire.subscribe(callback)
    try:
        yield spans
    finally:
        logfire.unsubscribe(token)


# Usage in tests:
# def test_instrumentation() -> None:
#     """Test that operations are properly instrumented."""
#     with capture_spans() as spans:
#         client.execute_sync("quick_review")

#     assert len(spans) > 0
#     assert any(s["name"] == "execute_task" for s in spans)
