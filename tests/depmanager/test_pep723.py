import pytest

from llmling.depmanager.exceptions import ScriptError
from llmling.depmanager.parser import parse_pep723_deps, parse_script_metadata


# Test data
SCRIPT_WITH_DEPS = """\
# /// script
# dependencies = [
#   "requests>=2.31.0",
#   "pandas>=2.0.0"
# ]
# requires-python = ">=3.12"
# ///

import requests
import pandas as pd
"""

SCRIPT_SINGLE_DEP = """\
# /// script
# dependencies = ["requests>=2.31.0"]
# ///
"""

SCRIPT_INVALID_TOML = """\
# /// script
# dependencies = ["incomplete
# ///
"""

SCRIPT_MULTIPLE_BLOCKS = """\
# /// script
# dependencies = ["pkg1"]
# ///

# /// script
# dependencies = ["pkg2"]
# ///
"""

SCRIPT_PYTHON_VERSION = """\
# /// script
# requires-python = ">=3.12"
# dependencies = ["requests"]
# ///
"""

SCRIPT_NO_DEPS = "print('hello')"


class TestPEP723Parser:
    def test_parse_deps(self) -> None:
        """Test parsing of PEP 723 dependencies."""
        deps = list(parse_pep723_deps(SCRIPT_WITH_DEPS))
        assert deps == ["requests>=2.31.0", "pandas>=2.0.0"]

    def test_parse_metadata(self) -> None:
        """Test parsing complete metadata."""
        metadata = parse_script_metadata(SCRIPT_WITH_DEPS)
        assert metadata.dependencies == ["requests>=2.31.0", "pandas>=2.0.0"]
        assert metadata.python_version == ">=3.12"

    def test_invalid_toml(self) -> None:
        """Test handling of invalid TOML in script metadata."""
        with pytest.raises(ScriptError, match="Invalid TOML"):
            parse_script_metadata(SCRIPT_INVALID_TOML)

    def test_multiple_metadata_blocks(self) -> None:
        """Test handling of multiple script metadata blocks."""
        with pytest.raises(ScriptError, match="Multiple script metadata blocks"):
            parse_script_metadata(SCRIPT_MULTIPLE_BLOCKS)

    def test_python_version_only(self) -> None:
        """Test parsing Python version without dependencies."""
        metadata = parse_script_metadata(SCRIPT_PYTHON_VERSION)
        assert metadata.dependencies == ["requests"]
        assert metadata.python_version == ">=3.12"

    def test_no_metadata_block(self) -> None:
        """Test file without metadata block."""
        metadata = parse_script_metadata(SCRIPT_NO_DEPS)
        assert not metadata.dependencies
        assert metadata.python_version is None
