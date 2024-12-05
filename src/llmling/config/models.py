"""Configuration models for LLMling."""

from __future__ import annotations

from collections.abc import Sequence as TypingSequence  # noqa: TC003
import inspect
import os  # noqa: TC003
from typing import Annotated, Any, Literal, Self
import warnings

import logfire
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
import upath
import yamling

from llmling import config_resources
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.core.typedefs import ProcessingStep  # noqa: TC001
from llmling.processors.base import ProcessorConfig  # noqa: TC001
from llmling.prompts.models import PromptType  # noqa: TC001
from llmling.resources.watching import WatchConfig  # noqa: TC001
from llmling.tools.toolsets import ToolSet
from llmling.utils import importing
from llmling.utils.importing import import_callable
from llmling.utils.paths import guess_mime_type


ResourceType = Literal["path", "text", "cli", "source", "callable", "image"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

logger = get_logger(__name__)


class Jinja2Config(BaseModel):
    """Configuration for Jinja2 environment.

    See: https://jinja.palletsprojects.com/en/3.1.x/api/#jinja2.Environment
    """

    # Template loading and discovery
    block_start_string: str = "{%"
    block_end_string: str = "%}"
    variable_start_string: str = "{{"
    variable_end_string: str = "}}"
    comment_start_string: str = "{#"
    comment_end_string: str = "#}"
    line_statement_prefix: str | None = None
    line_comment_prefix: str | None = None
    trim_blocks: bool = False
    lstrip_blocks: bool = False
    newline_sequence: Literal["\n", "\r\n", "\r"] = "\n"
    keep_trailing_newline: bool = False

    # Extensions and plugins
    extensions: list[str] = Field(default_factory=list)
    undefined: Literal["default", "strict", "debug", "chainable"] = "default"
    # Custom filters and tests
    filters: dict[str, str] = Field(default_factory=dict)
    tests: dict[str, str] = Field(default_factory=dict)
    globals: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    def create_environment_kwargs(self) -> dict[str, Any]:
        """Convert config to Jinja2 environment kwargs.

        Creates a dictionary of kwargs for jinja2.Environment with proper
        conversion of special values.

        Returns:
            Dict of kwargs for jinja2.Environment constructor

        Raises:
            ValueError: If filter or test imports fail
        """
        import jinja2

        from llmling.utils import importing

        # Start with basic string/bool config items
        kwargs = self.model_dump(exclude={"undefined", "filters", "tests"})

        # Convert undefined to proper class
        kwargs["undefined"] = {
            "default": jinja2.Undefined,
            "strict": jinja2.StrictUndefined,
            "debug": jinja2.DebugUndefined,
            "chainable": jinja2.ChainableUndefined,
        }[self.undefined]

        try:
            # Import filters and tests (must be callables)
            kwargs["filters"] = {
                name: importing.import_callable(path)
                for name, path in self.filters.items()
            }
            kwargs["tests"] = {
                name: importing.import_callable(path) for name, path in self.tests.items()
            }
        except Exception as exc:
            msg = f"Failed to import Jinja2 filters/tests: {exc}"
            raise ValueError(msg) from exc

        return kwargs


class GlobalSettings(BaseModel):
    """Global settings that apply to all components."""

    timeout: int = 30
    """Maximum time in seconds to wait for operations"""

    max_retries: int = 3
    """Maximum number of retries for failed operations"""

    requirements: list[str] = Field(default_factory=list)
    """List of package requirments for the functions used in this file."""

    pip_index_url: str | None = None
    """Alternative PyPI index URL for package installation"""

    extra_paths: list[str] = Field(default_factory=list)
    """Additional import paths"""

    scripts: list[str] = Field(default_factory=list)
    """PEP723 scripts (can be imported and will be scanned for dependencies)"""

    prefer_uv: bool = False
    """Explicitely use uv for package installation / management """

    log_level: LogLevel = "INFO"
    """Log level to use for the server"""

    jinja_environment: Jinja2Config = Field(default_factory=Jinja2Config)

    model_config = ConfigDict(frozen=True)


class BaseResource(BaseModel):
    """Base class for all resource types."""

    resource_type: str = Field(init=False)
    """Type identifier for this resource."""

    description: str = ""
    """Human-readable description of the resource."""

    uri: str | None = None
    """Canonical URI for this resource, set during registration if unset."""

    processors: list[ProcessingStep] = Field(default_factory=list)
    """Processing steps to apply when loading this resource."""

    watch: WatchConfig | None = None
    """Configuration for file system watching, if supported."""

    model_config = ConfigDict(frozen=True)

    @property
    def supports_watching(self) -> bool:
        """Whether this resource instance supports watching."""
        return False

    def is_watched(self) -> bool:
        """Tell if this resource should be watched."""
        return self.supports_watching and self.watch is not None and self.watch.enabled

    def is_templated(self) -> bool:
        """Whether this resource supports URI templates."""
        return False  # Default: resources are static

    @property
    def mime_type(self) -> str:
        """Get the MIME type for this resource.

        This should be overridden by subclasses that can determine
        their MIME type. Default is text/plain.
        """
        return "text/plain"


class PathResource(BaseResource):
    """Resource loaded from a file or URL."""

    resource_type: Literal["path"] = Field(default="path", init=False)
    path: str | os.PathLike[str]
    watch: WatchConfig | None = None

    @property
    def supports_watching(self) -> bool:
        """Whether this resource instance supports watching."""
        path = upath.UPath(self.path)
        if not path.exists():
            msg = f"Cannot watch non-existent path: {self.path}"
            warnings.warn(msg, UserWarning, stacklevel=2)
            return False
        return True

    @model_validator(mode="after")
    def validate_path(self) -> PathResource:
        """Validate that the path is not empty."""
        if not self.path:
            msg = "Path cannot be empty"
            raise ValueError(msg)
        return self

    def is_templated(self) -> bool:
        """Path resources are templated if they contain placeholders."""
        return "{" in str(self.path)

    @property
    def mime_type(self) -> str:
        """Get MIME type based on file extension."""
        return guess_mime_type(self.path)


class TextResource(BaseResource):
    """Raw text resource."""

    resource_type: Literal["text"] = Field(default="text", init=False)
    content: str

    @model_validator(mode="after")
    def validate_content(self) -> TextResource:
        """Validate that the content is not empty."""
        if not self.content:
            msg = "Content cannot be empty"
            raise ValueError(msg)
        return self

    _mime_type: str | None = None  # Optional override

    @property
    def mime_type(self) -> str:
        """Get MIME type, trying to detect JSON/YAML."""
        if self._mime_type:
            return self._mime_type
        # Could add content inspection here
        return "text/plain"


class CLIResource(BaseResource):
    """Resource from CLI command execution."""

    resource_type: Literal["cli"] = Field(default="cli", init=False)
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

    resource_type: Literal["source"] = Field(default="source", init=False)
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

    resource_type: Literal["callable"] = Field(default="callable", init=False)
    import_path: str
    keyword_args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_import_path(self) -> CallableResource:
        """Validate that the import path is properly formatted."""
        if not all(part.isidentifier() for part in self.import_path.split(".")):
            msg = f"Invalid import path: {self.import_path}"
            raise ValueError(msg)
        return self

    def is_templated(self) -> bool:
        """Callable resources are templated if they take parameters."""
        fn = import_callable(self.import_path)
        sig = inspect.signature(fn)
        return bool(sig.parameters)


class ImageResource(BaseResource):
    """Resource for image input."""

    resource_type: Literal["image"] = Field(default="image", init=False)
    path: str  # Local path or URL
    alt_text: str | None = None
    watch: WatchConfig | None = None

    model_config = ConfigDict(frozen=True)

    @property
    def supports_watching(self) -> bool:
        """Whether this resource instance supports watching."""
        return True

    @model_validator(mode="before")
    @classmethod
    def validate_path(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Validate that path is not empty."""
        if isinstance(data, dict) and not data.get("path"):
            msg = "Path cannot be empty for image resource"
            raise ValueError(msg)
        return data

    @property
    def mime_type(self) -> str:
        """Get MIME type based on file extension or default to image/jpeg."""
        try:
            mime = guess_mime_type(self.path)
            # If it's an image type, return it
            if mime.startswith("image/"):
                return mime
            # If not an image type or unknown, fall back to jpeg
        except Exception:  # noqa: BLE001
            return "image/jpeg"
        else:
            return "image/jpeg"


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


class BaseToolsetConfig(BaseModel):
    """Base configuration for toolsets."""

    namespace: str | None = Field(
        default=None, description="Optional namespace prefix for tool names"
    )


class OpenAPIToolsetConfig(BaseToolsetConfig):
    """Configuration for OpenAPI toolsets."""

    type: Literal["openapi"] = Field("openapi", init=False)
    spec: str = Field(..., description="URL or path to OpenAPI specification")
    base_url: str | None = Field(default=None, description="Base URL for API requests")


class EntryPointToolsetConfig(BaseToolsetConfig):
    """Configuration for entry point toolsets."""

    type: Literal["entry_points"] = Field("entry_points", init=False)
    module: str = Field(..., description="Python module path")


class CustomToolsetConfig(BaseToolsetConfig):
    """Configuration for custom toolsets."""

    type: Literal["custom"] = Field("custom", init=False)
    import_path: str = Field(..., description="Import path to toolset class")

    @field_validator("import_path", mode="after")
    @classmethod
    def validate_import_path(cls, v: str) -> str:
        # v is already confirmed to be a str here
        try:
            cls = importing.import_class(v)
            if not issubclass(cls, ToolSet):
                msg = f"{v} must be a ToolSet class"
                raise ValueError(msg)  # noqa: TRY004, TRY301
        except Exception as exc:
            msg = f"Invalid toolset class: {v}"
            raise ValueError(msg) from exc
        return v


# Use discriminated union for toolset types
ToolsetConfig = Annotated[
    OpenAPIToolsetConfig | EntryPointToolsetConfig | CustomToolsetConfig,
    Field(discriminator="type"),
]


class Config(BaseModel):
    """Root configuration model."""

    version: str = "1.0"
    global_settings: GlobalSettings = Field(default_factory=GlobalSettings)
    context_processors: dict[str, ProcessorConfig] = Field(default_factory=dict)
    resources: dict[str, Resource] = Field(default_factory=dict)
    resource_groups: dict[str, list[str]] = Field(default_factory=dict)
    tools: dict[str, ToolConfig] = Field(default_factory=dict)
    toolsets: dict[str, ToolsetConfig] = Field(default_factory=dict)
    prompts: dict[str, PromptType] = Field(default_factory=dict)

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def populate_prompt_names(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Populate prompt names from dictionary keys before validation."""
        if isinstance(data, dict) and "prompts" in data:
            prompts = data["prompts"]
            if isinstance(prompts, dict):
                # Add name to each prompt's data
                data["prompts"] = {
                    key: {
                        "name": key,
                        **(val if isinstance(val, dict) else val.model_dump()),
                    }
                    for key, val in prompts.items()
                }
        return data

    @model_validator(mode="after")
    def validate_references(self) -> Config:
        """Validate all references between components."""
        # Only validate if the optional components are present
        if self.resource_groups:
            self._validate_resource_groups()
        if self.context_processors:
            self._validate_processor_references()
        return self

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

    @classmethod
    @logfire.instrument("Loading configuration from {path}")
    def from_file(cls, path: str | os.PathLike[str]) -> Self:
        """Load configuration from YAML file.

        This function only handles the basic loading and model validation.
        For full validation and management, use ConfigManager.load() instead.

        Args:
            path: Path to configuration file

        Returns:
            Loaded configuration

        Raises:
            ConfigError: If loading fails
        """
        logger.debug("Loading configuration from %s", path)

        try:
            content = yamling.load_yaml_file(path)
        except Exception as exc:
            msg = f"Failed to load YAML from {path!r}"
            raise exceptions.ConfigError(msg) from exc

        # Validate basic structure
        if not isinstance(content, dict):
            msg = "Configuration must be a dictionary"
            raise exceptions.ConfigError(msg)

        try:
            config = cls.model_validate(content)
        except Exception as exc:
            msg = f"Failed to validate configuration from {path}"
            raise exceptions.ConfigError(msg) from exc
        else:
            msg = "Loaded raw configuration: version=%s, resources=%d"
            logger.debug(msg, config.version, len(config.resources))
            return config


if __name__ == "__main__":
    from llmling import Config

    config = Config.from_file(config_resources.TEST_CONFIG)  # type: ignore[has-type]
    print(config)
