"""Configuration models for LLMling."""

from __future__ import annotations

import os  # noqa: TC003
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
)
import upath
import yamling


class ConfigModel(BaseModel):
    """Base class for all LLMling configuration models.

    Provides:
    - Common Pydantic settings
    - YAML serialization
    - Basic merge functionality
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        use_attribute_docstrings=True,
    )

    def merge(self, other: Self) -> Self:
        """Merge with another instance by overlaying its non-None values."""
        from llmling.config.utils import merge_models

        return merge_models(self, other)

    @classmethod
    def from_yaml(
        cls,
        content: str,
        inherit_path: str | os.PathLike[str] | None = None,
    ) -> Self:
        """Create from YAML string."""
        data = yamling.load_yaml(content, resolve_inherit=inherit_path or False)
        return cls.model_validate(data)

    def model_dump_yaml(self) -> str:
        """Dump configuration to YAML string."""
        import yamling

        return yamling.dump_yaml(self.model_dump(exclude_none=True))

    def save(self, path: str | os.PathLike[str], overwrite: bool = False) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to save the configuration to
            overwrite: Whether to overwrite an existing file

        Raises:
            OSError: If file cannot be written
            ValueError: If path is invalid
        """
        try:
            yaml_str = self.model_dump_yaml()
            file_path = upath.UPath(path)
            if file_path.exists() and not overwrite:
                msg = f"File already exists: {path}"
                raise FileExistsError(msg)  # noqa: TRY301
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(yaml_str)
        except Exception as exc:
            msg = f"Failed to save configuration to {path}"
            raise ValueError(msg) from exc
