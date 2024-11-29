"""Functions for tests.

We need some functions importable during the tests for proper testing.
"""

from llmling.testing.processors import (
    reverse_text,
    uppercase_text,
    multiply,
    append_text,
    async_reverse,
    async_reverse_text,
    failing_processor,
)
from llmling.testing.tools import (
    failing_tool,
    example_tool,
    analyze_ast,
)
from llmling.testing.utils import TestStreamPair, create_test_server_session


def get_mcp_tools():
    """Entry point exposing test tools to LLMling."""
    from llmling.testing.tools import example_tool, analyze_ast

    return [example_tool, analyze_ast]


__all__ = [
    # Test utilities
    "TestStreamPair",
    "analyze_ast",
    "append_text",
    "async_reverse",
    "async_reverse_text",
    "create_test_server_session",
    "example_tool",
    "failing_processor",
    # Tools
    "failing_tool",
    "multiply",
    # Processors
    "reverse_text",
    "uppercase_text",
]
