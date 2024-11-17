"""Configuration models for LLM management system."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

import pydantic
from pydantic import BaseModel
import yamling

from llmling.processors import ProcessingStep  # noqa: TCH001


if TYPE_CHECKING:
    import os


MODEL_FORMAT_ERROR = "Model {model} must be in format 'provider/model'"
IMPORT_PATH_ERROR = "Invalid import path: {path}"
PROCESSOR_ERROR = "Processor {name} not found"
PROVIDER_REF_ERROR = "Provider {name} referenced in group {group} not found"
CONTEXT_REF_ERROR = "Context {name} referenced in group {group} not found"
TASK_PROVIDER_ERROR = "Provider {name} in task {task} not found"
TASK_CONTEXT_ERROR = "Context {name} in task {task} not found"


class GlobalSettings(BaseModel):
    """Global settings that apply to all LLMs unless overridden."""

    timeout: int = 30
    max_retries: int = 3
    temperature: float = 0.7


class ProcessorConfig(BaseModel):
    """Configuration for a text processor."""

    type: Literal["function", "template"]
    import_path: str | None = None
    template: str | None = None

    @pydantic.model_validator(mode="after")
    def validate_config(self) -> ProcessorConfig:
        """Validate processor configuration based on type."""
        match self.type:
            case "function" if not self.import_path:
                msg = "import_path is required for function processors"
                raise ValueError(msg)
            case "template" if not self.template:
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
            msg = MODEL_FORMAT_ERROR.format(model=self.model)
            raise ValueError(msg)
        return self


class BaseContext(BaseModel):
    """Base class for all context types."""

    description: str
    processors: list[ProcessingStep] = []


class PathContext(BaseContext):
    """Context loaded from a file or URL."""

    type: Literal["path"]
    path: Annotated[str, pydantic.HttpUrl | pydantic.FilePath]


class TextContext(BaseContext):
    """Raw text context."""

    type: Literal["text"]
    content: str


class CLIContext(BaseContext):
    """Context from CLI command execution."""

    type: Literal["cli"]
    command: str | list[str]
    shell: bool = False
    cwd: str | None = None
    timeout: int | None = None


class SourceContext(BaseContext):
    """Context from Python source code."""

    type: Literal["source"]
    import_path: str
    recursive: bool = False
    include_tests: bool = False

    @pydantic.model_validator(mode="after")
    def validate_import_path(self) -> SourceContext:
        """Validate that the import path is properly formatted."""
        if not all(part.isidentifier() for part in self.import_path.split(".")):
            msg = IMPORT_PATH_ERROR.format(path=self.import_path)
            raise ValueError(msg)
        return self


class CallableContext(BaseContext):
    """Context from executing a Python callable."""

    type: Literal["callable"]
    import_path: str
    keyword_args: dict[str, Any] = {}

    @pydantic.model_validator(mode="after")
    def validate_import_path(self) -> CallableContext:
        """Validate that the import path is properly formatted."""
        if not all(part.isidentifier() for part in self.import_path.split(".")):
            msg = IMPORT_PATH_ERROR.format(path=self.import_path)
            raise ValueError(msg)
        return self


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
                    msg = PROVIDER_REF_ERROR.format(name=provider, group=group)
                    raise ValueError(msg)

    def _validate_context_groups(self) -> None:
        """Validate context references in groups."""
        for group, contexts in self.context_groups.items():
            for context in contexts:
                if context not in self.contexts:
                    msg = CONTEXT_REF_ERROR.format(name=context, group=group)
                    raise ValueError(msg)

    def _validate_processor_references(self) -> None:
        """Validate processor references in contexts."""
        all_processors = set(self.context_processors)
        for context in self.contexts.values():
            for step in context.processors:
                if step.name not in all_processors:
                    msg = PROCESSOR_ERROR.format(name=step.name)
                    raise ValueError(msg)

    def _validate_task_templates(self) -> None:
        """Validate task template references."""
        for name, task in self.task_templates.items():
            if (
                task.provider not in self.llm_providers
                and task.provider not in self.provider_groups
            ):
                msg = TASK_PROVIDER_ERROR.format(name=task.provider, task=name)
                raise ValueError(msg)

            if (
                task.context not in self.contexts
                and task.context not in self.context_groups
            ):
                msg = TASK_CONTEXT_ERROR.format(name=task.context, task=name)
                raise ValueError(msg)


def load_config(path: str | os.PathLike[str]) -> Config:
    """Load and validate configuration from YAML file."""
    content = yamling.load_yaml_file(path)
    return Config.model_validate(content)


if __name__ == "__main__":
    config = load_config("src/llmling/resources/test.yml")
    print(f"Version: {config.version}")
    print(f"Number of providers: {len(config.llm_providers)}")
    print(f"Number of contexts: {len(config.contexts)}")

    if gpt4_config := config.llm_providers.get("gpt4-turbo"):
        print(f"GPT-4 model: {gpt4_config.model}")

    source_contexts = [
        ctx for ctx in config.contexts.values() if isinstance(ctx, SourceContext)
    ]
    print(f"Number of source contexts: {len(source_contexts)}")
