"""Configuration models for LLMling."""

from __future__ import annotations

from collections.abc import Sequence as TypingSequence  # noqa: TC003
import os  # noqa: TC003
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmling.core.typedefs import ProcessingStep  # noqa: TC001
from llmling.processors.base import ProcessorConfig  # noqa: TC001
from llmling.prompts.models import Prompt  # noqa: TC001


ResourceType = Literal["path", "text", "cli", "source", "callable", "image"]


class GlobalSettings(BaseModel):
    """Global settings that apply to all components."""

    timeout: int = 30
    """Maximum time in seconds to wait for operations"""

    max_retries: int = 3
    """Maximum number of retries for failed operations"""

    temperature: float = 0.7
    """Default sampling temperature for LLM completions"""

    model_config = ConfigDict(frozen=True)


class BaseResource(BaseModel):
    """Base class for all resource types."""

    resource_type: ResourceType = Field(...)
    description: str = ""
    processors: list[ProcessingStep] = Field(
        default_factory=list
    )  # Optional with empty default
    model_config = ConfigDict(frozen=True)


class PathResource(BaseResource):
    """Resource loaded from a file or URL."""

    resource_type: Literal["path"] = "path"
    path: str | os.PathLike[str]

    @model_validator(mode="after")
    def validate_path(self) -> PathResource:
        """Validate that the path is not empty."""
        if not self.path:
            msg = "Path cannot be empty"
            raise ValueError(msg)
        return self


class TextResource(BaseResource):
    """Raw text resource."""

    resource_type: Literal["text"] = "text"
    content: str

    @model_validator(mode="after")
    def validate_content(self) -> TextResource:
        """Validate that the content is not empty."""
        if not self.content:
            msg = "Content cannot be empty"
            raise ValueError(msg)
        return self


class CLIResource(BaseResource):
    """Resource from CLI command execution."""

    resource_type: Literal["cli"] = "cli"
    command: str | TypingSequence[str]
    shell: bool = False
    cwd: str | None = None
    timeout: float | None = None

    @model_validator(mode="after")
    def validate_command(self) -> CLIResource:
        """Validate command configuration."""
        if not self.command:
            msg = "Command cannot be empty"
            raise ValueError(msg)
        if (
            isinstance(self.command, list | tuple)
            and not self.shell
            and not all(isinstance(part, str) for part in self.command)
        ):
            msg = "When shell=False, all command parts must be strings"
            raise ValueError(msg)
        return self


class SourceResource(BaseResource):
    """Resource from Python source code."""

    resource_type: Literal["source"] = "source"
    import_path: str
    recursive: bool = False
    include_tests: bool = False

    @model_validator(mode="after")
    def validate_import_path(self) -> SourceResource:
        """Validate that the import path is properly formatted."""
        if not all(part.isidentifier() for part in self.import_path.split(".")):
            msg = f"Invalid import path: {self.import_path}"
            raise ValueError(msg)
        return self


class CallableResource(BaseResource):
    """Resource from executing a Python callable."""

    resource_type: Literal["callable"] = "callable"
    import_path: str
    keyword_args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_import_path(self) -> CallableResource:
        """Validate that the import path is properly formatted."""
        if not all(part.isidentifier() for part in self.import_path.split(".")):
            msg = f"Invalid import path: {self.import_path}"
            raise ValueError(msg)
        return self


class ImageResource(BaseResource):
    """Resource for image input."""

    resource_type: Literal["image"] = "image"
    path: str  # Local path or URL
    alt_text: str | None = None

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="before")
    @classmethod
    def validate_path(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Validate that path is not empty."""
        if isinstance(data, dict) and not data.get("path"):
            msg = "Path cannot be empty for image resource"
            raise ValueError(msg)
        return data


Resource = (
    PathResource
    | TextResource
    | CLIResource
    | SourceResource
    | CallableResource
    | ImageResource
)


class ToolConfig(BaseModel):
    """Configuration for a tool."""

    import_path: str
    """Import path to the tool implementation (e.g. 'mymodule.tools.MyTool')"""

    name: str | None = None
    """Optional override for the tool's display name"""

    description: str | None = None
    """Optional override for the tool's description"""

    model_config = ConfigDict(frozen=True)


class Config(BaseModel):
    """Root configuration model."""

    version: str = "1.0"
    global_settings: GlobalSettings = Field(default_factory=GlobalSettings)
    context_processors: dict[str, ProcessorConfig] = Field(default_factory=dict)
    resources: dict[str, Resource] = Field(default_factory=dict)
    resource_groups: dict[str, list[str]] = Field(default_factory=dict)
    tools: dict[str, ToolConfig] = Field(default_factory=dict)
    # Add prompts support
    prompts: dict[str, Prompt] = Field(default_factory=dict)

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def validate_references(self) -> Config:
        """Validate all references between components."""
        # Only validate if the optional components are present
        if self.resource_groups:
            self._validate_resource_groups()
        if self.context_processors:
            self._validate_processor_references()
        if self.prompts:
            self._validate_prompts()
        return self

    def _validate_prompts(self) -> None:
        """Validate prompt configurations."""
        for name, prompt in self.prompts.items():
            if prompt.name != name:
                msg = f"Prompt name mismatch: {prompt.name} != {name}"
                raise ValueError(msg)

    def _validate_resource_groups(self) -> None:
        """Validate resource references in groups."""
        for group, resources in self.resource_groups.items():
            for resource in resources:
                if resource not in self.resources:
                    msg = f"Resource {resource} referenced in group {group} not found"
                    raise ValueError(msg)

    def _validate_processor_references(self) -> None:
        """Validate processor references in resources."""
        for resource in self.resources.values():
            for processor in resource.processors:
                if processor.name not in self.context_processors:
                    msg = f"Processor {processor.name!r} not found"
                    raise ValueError(msg)

    def model_dump_yaml(self) -> str:
        """Dump configuration to YAML string."""
        import yamling

        return yamling.dump_yaml(self.model_dump(exclude_none=True))


if __name__ == "__main__":
    from pydantic import ValidationError

    from llmling.config.loading import load_config

    try:
        config = load_config("src/llmling/config_resources/test.yml")  # type: ignore[has-type]
        print(config)
    except ValidationError as e:
        print(e)
