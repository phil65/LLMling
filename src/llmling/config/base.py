"""Configuration models for LLMling."""

from pydantic import ConfigDict
from schemez import Schema


class ConfigModel(Schema):
    """Base class for all LLMling configuration models.

    Provides:
    - Common Pydantic settings
    - YAML serialization
    - Basic merge functionality
    """

    model_config = ConfigDict(frozen=True)
