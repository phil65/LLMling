"""Configuration models for LLM management system."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

import pydantic
from pydantic import BaseModel
import yamling

from llmling.processors import ProcessingStep  # noqa: TCH001


if TYPE_CHECKING:
    import os


class GlobalSettings(BaseModel):
    """Global settings that apply to all LLMs unless overridden."""

    timeout: int = 30
    max_retries: int = 3
    temperature: float = 0.7


class ProcessorConfig(BaseModel):
    """Configuration for a text processor."""

    type: Literal["function", "template"]
    import_path: str | None = None  # Required for function type
    template: str | None = None  # Required for template type

    @pydantic.model_validator(mode="after")
    def validate_config(self) -> ProcessorConfig:
        """Validate processor configuration based on type."""
        if self.type == "function" and not self.import_path:
            msg = "import_path is required for function processors"
            raise ValueError(msg)
        if self.type == "template" and not self.template:
            msg = "template is required for template processors"
            raise ValueError(msg)
        return self


class LLMProvider(BaseModel):
    """LLM provider configuration."""

    model: str  # litellm format: provider/model
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None

    @pydantic.model_validator(mode="after")
    def validate_model_format(self) -> LLMProvider:
        """Validate that model follows provider/name format."""
        if "/" not in self.model:
            msg = f"Model {self.model} must be in format 'provider/model'"
            raise ValueError(msg)
        return self


class PathContext(BaseModel):
    """Context loaded from a file or URL."""

    type: Literal["path"]
    path: Annotated[str, pydantic.HttpUrl | pydantic.FilePath]
    description: str
    processors: list[ProcessingStep] = []


class TextContext(BaseModel):
    """Raw text context."""

    type: Literal["text"]
    content: str
    description: str
    processors: list[ProcessingStep] = []


class CLIContext(BaseModel):
    """Context from CLI command execution."""

    type: Literal["cli"]
    command: str | list[str]
    description: str
    shell: bool = False
    cwd: str | None = None
    timeout: int | None = None
    processors: list[ProcessingStep] = []


class SourceContext(BaseModel):
    """Context from Python source code."""

    type: Literal["source"]
    import_path: str
    description: str
    recursive: bool = False
    include_tests: bool = False
    processors: list[ProcessingStep] = []

    @pydantic.model_validator(mode="after")
    def validate_import_path(self) -> SourceContext:
        """Validate that the import path is properly formatted."""
        if not all(part.isidentifier() for part in self.import_path.split(".")):
            msg = f"Invalid import path: {self.import_path}"
            raise ValueError(msg)
        return self


class CallableContext(BaseModel):
    """Context from executing a Python callable."""

    type: Literal["callable"]
    import_path: str
    description: str
    keyword_args: dict[str, Any] = {}
    processors: list[ProcessingStep] = []

    @pydantic.model_validator(mode="after")
    def validate_import_path(self) -> CallableContext:
        """Validate that the import path is properly formatted."""
        if not all(part.isidentifier() for part in self.import_path.split(".")):
            msg = f"Invalid import path: {self.import_path}"
            raise ValueError(msg)
        return self


# Update the Context type to include CallableContext
Context = PathContext | TextContext | CLIContext | SourceContext | CallableContext


class TaskSettings(BaseModel):
    """Settings for a specific task."""

    temperature: float | None = None
    max_tokens: int | None = None


class TaskTemplate(BaseModel):
    """Template for a specific task."""

    provider: str  # either provider name or group name
    context: str  # either context name or group name
    settings: TaskSettings
    inherit_tools: bool = True


class Config(BaseModel):
    """Root configuration model."""

    version: str
    global_settings: GlobalSettings
    context_processors: dict[str, ProcessorConfig]
    llm_providers: dict[str, LLMProvider]
    provider_groups: dict[str, list[str]]
    contexts: dict[str, Context]
    context_groups: dict[str, list[str]]
    task_templates: dict[str, TaskTemplate]

    @pydantic.model_validator(mode="after")
    def validate_references(self) -> Config:
        """Validate all references between components."""
        # Validate provider references in groups
        for group, providers in self.provider_groups.items():
            for provider in providers:
                if provider not in self.llm_providers:
                    msg = f"Provider {provider} referenced in group {group} not found"
                    raise ValueError(msg)

        # Validate context references in groups
        for group, contexts in self.context_groups.items():
            for context in contexts:
                if context not in self.contexts:
                    msg = f"Context {context} referenced in group {group} not found"
                    raise ValueError(msg)

        # Validate processor references in contexts
        all_processors = set(self.context_processors.keys())
        for ctx in self.contexts.values():
            for step in ctx.processors:
                if step.name not in all_processors:
                    msg = f"Processor {step.name} not found"
                    raise ValueError(msg)

        # Validate task template references
        for name, task in self.task_templates.items():
            # Check provider reference
            if (
                task.provider not in self.llm_providers
                and task.provider not in self.provider_groups
            ):
                msg = f"Provider {task.provider} in task {name} not found"
                raise ValueError(msg)

            # Check context reference
            if (
                task.context not in self.contexts
                and task.context not in self.context_groups
            ):
                msg = f"Context {task.context} in task {name} not found"
                raise ValueError(msg)

        return self


def load_config(path: str | os.PathLike[str]) -> Config:
    """Load and validate configuration from YAML file."""
    content = yamling.load_yaml_file(path)
    return Config.model_validate(content)


# Usage example:
if __name__ == "__main__":
    # Load configuration
    config = load_config("src/llmling/resources/test.yml")

    # Access configuration
    print(f"Version: {config.version}")
    print(f"Number of providers: {len(config.llm_providers)}")
    print(f"Number of contexts: {len(config.contexts)}")

    # Validate specific provider
    gpt4_config = config.llm_providers.get("gpt4-turbo")
    if gpt4_config:
        print(f"GPT-4 model: {gpt4_config.model}")

    # Get all contexts of a specific type
    source_contexts = [
        ctx for ctx in config.contexts.values() if isinstance(ctx, SourceContext)
    ]
    print(f"Number of source contexts: {len(source_contexts)}")
