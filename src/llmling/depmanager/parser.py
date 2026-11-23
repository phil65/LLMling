"""PEP 723 dependency parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import sys
import tomllib
from typing import TYPE_CHECKING

from packaging import specifiers, version

from llmling.depmanager.exceptions import DependencyError, ScriptError


if TYPE_CHECKING:
    from collections.abc import Iterator


# PEP 723 regex pattern
SCRIPT_REGEX = (
    r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s"
    r"(?P<content>(^#(| .*)$\s)+)^# ///$"
)


logger = logging.getLogger(__name__)


@dataclass
class ScriptMetadata:
    """Metadata extracted from a script."""

    dependencies: list[str]
    python_version: str | None = None


def parse_script_metadata(content: str) -> ScriptMetadata:
    """Parse PEP 723 metadata from Python content.

    Format:
        # /// script
        # dependencies = [
        #   "requests<3",
        #   "rich",
        # ]
        # requires-python = ">=3.11"
        # ///

    Args:
        content: Python source code content

    Returns:
        ScriptMetadata containing dependencies and Python version requirement

    Raises:
        ScriptError: If the script metadata is invalid or malformed
    """

    def extract_toml(match: re.Match[str]) -> str:
        """Extract TOML content from comment block."""
        lines = match.group("content").splitlines(keepends=True)
        return "".join(line[2:] if line.startswith("# ") else line[1:] for line in lines)

    # Find script metadata blocks
    matches = list(
        filter(lambda m: m.group("type") == "script", re.finditer(SCRIPT_REGEX, content))
    )

    if len(matches) > 1:
        msg = "Multiple script metadata blocks found"
        raise ScriptError(msg)

    if not matches:
        # No metadata block found
        return ScriptMetadata(dependencies=[])

    try:
        # Parse TOML content
        toml_content = extract_toml(matches[0])
        metadata = tomllib.loads(toml_content)

        # Handle dependencies
        deps = metadata.get("dependencies", [])
        if not isinstance(deps, list):
            msg = "dependencies must be a list"
            raise ScriptError(msg)  # noqa: TRY301

        # Handle Python version
        python_req = metadata.get("requires-python")
        if python_req is not None and not isinstance(python_req, str):
            msg = "requires-python must be a string"
            raise ScriptError(msg)  # noqa: TRY301

        return ScriptMetadata(dependencies=deps, python_version=python_req)

    except tomllib.TOMLDecodeError as exc:
        msg = f"Invalid TOML in script metadata: {exc}"
        raise ScriptError(msg) from exc
    except Exception as exc:
        msg = f"Error parsing script metadata: {exc}"
        raise ScriptError(msg) from exc


def parse_pep723_deps(content: str) -> Iterator[str]:
    """Parse dependency declarations from Python content.

    Backwards compatible interface for getting just the dependencies.

    Args:
        content: Python source code content

    Yields:
        Dependency specifications
    """
    metadata = parse_script_metadata(content)
    yield from metadata.dependencies


def check_python_version(version_spec: str, script_path: str) -> None:
    """Check if current Python version matches the requirement.

    Args:
        version_spec: PEP 440 version specifier (e.g., ">=3.12")
        script_path: Path to script (for error messages)

    Raises:
        DependencyError: If Python version doesn't match requirement
    """
    try:
        spec = specifiers.SpecifierSet(version_spec)
        python_version = version.Version(sys.version.split()[0])

        if python_version not in spec:
            msg = (
                f"Script {script_path} requires Python {version_spec}, "
                f"but current version is {python_version}"
            )
            raise DependencyError(msg)  # noqa: TRY301

    except Exception as exc:
        if isinstance(exc, DependencyError):
            raise
        msg = f"Invalid Python version specifier {version_spec!r}: {exc}"
        raise DependencyError(msg) from exc
