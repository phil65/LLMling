"""Utility functions for the config package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling.config.models import (
    CustomToolsetConfig,
    EntryPointToolsetConfig,
    OpenAPIToolsetConfig,
)
from llmling.tools.entry_points import EntryPointTools
from llmling.tools.openapi import OpenAPITools
from llmling.utils import importing


if TYPE_CHECKING:
    from llmling.tools.toolsets import ToolSet


def toolset_config_to_toolset(config) -> ToolSet:
    match config:
        case OpenAPIToolsetConfig():
            return OpenAPITools(
                spec=config.spec,
                base_url=config.base_url or "",
            )
        case EntryPointToolsetConfig():
            return EntryPointTools(config.module)
        case CustomToolsetConfig():
            toolset_class = importing.import_class(config.import_path)
            return toolset_class()
        case _:
            msg = f"Unknown toolset type: {type(config)}"
            raise ValueError(msg)
