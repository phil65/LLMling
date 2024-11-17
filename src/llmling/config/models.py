"""Configuration models for LLMling."""

from __future__ import annotations

from collections.abc import Sequence as TypingSequence  # noqa: TCH003
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmling.core.typedefs import ProcessingStep  # noqa: TCH001
from llmling.processors.base import ProcessorConfig  # noqa: TCH001


class GlobalSettings(BaseModel):
    """Global settings that apply to all components."""

    timeout: int = 30
    max_retries: int = 3
    temperature: float = 0.7
    model_config = ConfigDict(frozen=True)


class LLMProviderConfig(BaseModel):
    """LLM provider configuration."""

    name: str = ""  # Add this field
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_model_format(self) -> LLMProviderConfig:
        """Validate that model follows provider/name format."""
        if "/" not in self.model:
            msg = f"Model {self.model} must be in format 'provider/model'"
            raise ValueError(msg)
        return self


class TaskSettings(BaseModel):
    """Settings for a specific task."""

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None

    model_config = ConfigDict(frozen=True)


class BaseContext(BaseModel):
    """Base class for all context types."""

    type: str
    description: str
    processors: list[ProcessingStep] = Field(default_factory=list)
    model_config = ConfigDict(frozen=True)


class PathContext(BaseContext):
    """Context loaded from a file or URL."""

    type: Literal["path"]
    path: str

    @model_validator(mode="after")
    def validate_path(self) -> PathContext:
        """Validate that the path is not empty."""
        if not self.path:
            msg = "Path cannot be empty"
            raise ValueError(msg)
        return self


class TextContext(BaseContext):
    """Raw text context."""

    type: Literal["text"]
    content: str

    @model_validator(mode="after")
    def validate_content(self) -> TextContext:
        """Validate that the content is not empty."""
        if not self.content:
            msg = "Content cannot be empty"
            raise ValueError(msg)
        return self


class CLIContext(BaseContext):
    """Context from CLI command execution."""

    type: Literal["cli"]
    command: str | TypingSequence[str]
    shell: bool = False
    cwd: str | None = None
    timeout: float | None = None

    @model_validator(mode="after")
    def validate_command(self) -> CLIContext:
        """Validate command configuration."""
        if not self.command:
            msg = "Command cannot be empty"
            raise ValueError(msg)
        if isinstance(self.command, (list, tuple)) and not self.shell:
            for part in self.command:
                if not isinstance(part, str):
                    msg = "Command parts must be strings"
                    raise ValueError(msg)
        return self


class SourceContext(BaseContext):
    """Context from Python source code."""

    type: Literal["source"]
    import_path: str
    recursive: bool = False
    include_tests: bool = False

    @model_validator(mode="after")
    def validate_import_path(self) -> SourceContext:
        """Validate that the import path is properly formatted."""
        if not all(part.isidentifier() for part in self.import_path.split(".")):
            msg = f"Invalid import path: {self.import_path}"
            raise ValueError(msg)
        return self


class CallableContext(BaseContext):
    """Context from executing a Python callable."""

    type: Literal["callable"]
    import_path: str
    keyword_args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_import_path(self) -> CallableContext:
        """Validate that the import path is properly formatted."""
        if not all(part.isidentifier() for part in self.import_path.split(".")):
            msg = f"Invalid import path: {self.import_path}"
            raise ValueError(msg)
        return self


Context = PathContext | TextContext | CLIContext | SourceContext | CallableContext


class TaskTemplate(BaseModel):
    """Template for a specific task."""

    provider: str  # provider name or group name
    context: str  # context name or group name
    settings: TaskSettings | None = None
    inherit_tools: bool = True

    model_config = ConfigDict(frozen=True)


class Config(BaseModel):
    """Root configuration model."""

    version: str
    global_settings: GlobalSettings
    context_processors: dict[str, ProcessorConfig]
    llm_providers: dict[str, LLMProviderConfig]
    provider_groups: dict[str, list[str]] = Field(default_factory=dict)
    contexts: dict[str, Context]
    context_groups: dict[str, list[str]] = Field(default_factory=dict)
    task_templates: dict[str, TaskTemplate]

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def validate_references(self) -> Config:
        """Validate all references between components."""
        self._validate_provider_groups()
        self._validate_context_groups()
        self._validate_processor_references()
        self._validate_task_templates()
        return self

    def _validate_provider_groups(self) -> None:
        """Validate provider references in groups."""
        for group, providers in self.provider_groups.items():
            for provider in providers:
                if provider not in self.llm_providers:
                    msg = f"Provider {provider} referenced in group {group} not found"
                    raise ValueError(msg)

    def _validate_context_groups(self) -> None:
        """Validate context references in groups."""
        for group, contexts in self.context_groups.items():
            for context in contexts:
                if context not in self.contexts:
                    msg = f"Context {context} referenced in group {group} not found"
                    raise ValueError(msg)

    def _validate_processor_references(self) -> None:
        """Validate processor references in contexts."""
        for context in self.contexts.values():
            for processor in context.processors:
                if processor.name not in self.context_processors:
                    msg = f"Processor {processor.name!r} not found"
                    raise ValueError(msg)

    def _validate_task_templates(self) -> None:
        """Validate task template references."""
        for name, template in self.task_templates.items():
            # Validate provider reference
            if (
                template.provider not in self.llm_providers
                and template.provider not in self.provider_groups
            ):
                msg = f"Provider {template.provider} referenced in task {name} not found"
                raise ValueError(msg)

            # Validate context reference
            if (
                template.context not in self.contexts
                and template.context not in self.context_groups
            ):
                msg = f"Context {template.context} referenced in task {name} not found"
                raise ValueError(msg)


if __name__ == "__main__":
    from pydantic import ValidationError

    from llmling.config.loading import load_config

    try:
        config = load_config("src/llmling/resources/test.yml")
        print(config)
    except ValidationError as e:
        print(e)
