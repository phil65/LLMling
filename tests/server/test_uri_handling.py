from __future__ import annotations

from mcp.types import AnyUrl
import pytest

from llmling.server.conversions import from_mcp_uri, to_mcp_uri


@pytest.mark.parametrize(
    ("internal_uri", "mcp_uri"),
    [
        # File URIs
        ("file:///path/to/file.txt", "file://host/path/to/file.txt"),
        (
            "file:///C:/path/to/file.txt",
            "file://host/c/path/to/file.txt",
        ),  # Convert to /c/path format
        # HTTP URIs
        ("http://example.com", "http://example.com/"),
        ("https://example.com", "https://example.com/"),
        # Resource URIs
        ("text://content", "resource://host/content"),
        ("python://module.name", "resource://host/module.name"),
        ("cli://command", "resource://host/command"),
    ],
)
def test_uri_conversion_roundtrip(internal_uri: str, mcp_uri: str) -> None:
    """Test URI conversion in both directions."""
    # Internal -> MCP
    assert str(to_mcp_uri(internal_uri)) == mcp_uri
    # MCP -> Internal (for supported schemes)
    if not internal_uri.startswith(("text://", "python://", "cli://")):
        assert from_mcp_uri(mcp_uri) == internal_uri


@pytest.mark.parametrize(
    "uri",
    [
        "invalid://uri",
        "unknown://test",
        "://incomplete",
        "",  # Empty URI
    ],
)
def test_invalid_uri_conversion(uri: str) -> None:
    """Test handling of invalid URIs."""
    with pytest.raises(ValueError, match="URI"):
        to_mcp_uri(uri)
    with pytest.raises(ValueError, match="URI"):
        from_mcp_uri(uri)


def test_uri_type_handling() -> None:
    """Test handling of different URI input types."""
    # Test AnyUrl input
    mcp_uri = AnyUrl("file://host/test.txt")
    assert from_mcp_uri(str(mcp_uri)) == "file:///test.txt"

    # Test conversion to AnyUrl
    internal_uri = "file:///test.txt"
    assert isinstance(to_mcp_uri(internal_uri), AnyUrl)
