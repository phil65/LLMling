"""Path handling utilities for URI and filesystem operations."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
import urllib.parse


if TYPE_CHECKING:
    from collections.abc import Sequence


INVALID_CHARS_PATTERN = re.compile(r'[\x00-\x1F<>:"|?*\\]')


def is_ignorable_path_component(part: str) -> bool:
    """Check if a path component should be ignored."""
    return (
        not part
        or part in {".", ".."}
        or (len(part) == 2 and part[1] == ":")  # Drive letter  # noqa: PLR2004
        or part in {"/", "\\"}
    )


def normalize_path_components(parts: Sequence[str]) -> list[str]:
    """Normalize path components, resolving . and .. entries."""
    result: list[str] = []
    for part in parts:
        match part:
            case "." | "":
                continue
            case "..":
                if result:
                    result.pop()
            case _:
                if not is_ignorable_path_component(part):
                    result.append(part)
    return result


def is_ignorable_part(part: str) -> bool:
    """Check if a path component should be ignored."""
    return (
        not part
        or part in {".", ".."}
        or (len(part) == 2 and part[1] == ":")  # Drive letter  # noqa: PLR2004
        or part in {"/", "\\"}
    )


def uri_to_path(uri: str) -> str:
    """Convert a file URI to a normalized path."""
    if not uri.startswith("file:///"):
        msg = f"Invalid file URI format: {uri}"
        raise ValueError(msg)

    # Split into components and decode, including empty parts
    parts = [urllib.parse.unquote(part) for part in uri[8:].split("/")]

    # Normalize components
    return "/".join(normalize_path_components(parts))


def path_to_uri(path: str) -> str:
    """Convert a path to a file URI with proper encoding."""
    # Normalize path separators and split
    parts = path.replace("\\", "/").split("/")

    # Normalize and encode components
    normalized = normalize_path_components(parts)
    if not normalized:
        msg = "Empty path"
        raise ValueError(msg)

    encoded = [urllib.parse.quote(part) for part in normalized]
    return f"file:///{'/'.join(encoded)}"
