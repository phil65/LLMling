from __future__ import annotations

from llmling.extensions.loaders import ToolsetLoader


def test_toolset_loader():
    """Test loading tools from entry points."""
    loader = ToolsetLoader()
    tools = loader.load_items(["llmling"])  # just the package name
    assert "example_tool" in tools
    assert "analyze_ast" in tools
