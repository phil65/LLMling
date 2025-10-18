"""Configuration models for LLMling."""

from __future__ import annotations

from collections.abc import Sequence as TypingSequence
import inspect
import os
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self
import warnings

from pydantic import (
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from schemez import Schema

from llmling import config_resources
from llmling.config.base import ConfigModel
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.core.typedefs import ProcessingStep
from llmling.processors.base import ProcessorConfig  # noqa: TC001
from llmling.prompts.models import PromptType  # noqa: TC001
from llmling.utils.importing import import_callable, import_class
from llmling.utils.paths import guess_mime_type


if TYPE_CHECKING:
    from upath.types import JoinablePathLike


ResourceType = Literal["path", "text", "cli", "source", "callable"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

logger = get_logger(__name__)


class LLMCapabilitiesConfig(ConfigModel):
    """Configuration for LLM system capabilities."""

    load_resource: bool = False
    """Whether the LLM can load and access resource content."""

    get_resources: bool = False
    """Whether the LLM can discover available resources."""

    install_package: bool = False
    """Whether the LLM can install new Python packages for future tools."""

    register_tool: bool = False
    """Whether the LLM can register importable functions as new tools."""

    register_code_tool: bool = False
    """Whether the LLM can create new tools from provided Python code."""

    chain_tools: bool = False
    """Whether the LLM gains capability to chain multiple tool calls into one."""


class Jinja2Config(ConfigModel):
    """Configuration for Jinja2 environment.

    See: https://jinja.palletsprojects.com/en/3.1.x/api/#jinja2.Environment
    """

    block_start_string: str = "{%"
    """String denoting the beginning of a block (default: '{%')."""

    block_end_string: str = "%}"
    """String denoting the end of a block (default: '%}')."""

    variable_start_string: str = "{{"
    """String denoting the beginning of a variable (default: '{{')."""

    variable_end_string: str = "}}"
    """String denoting the end of a variable (default: '}}')."""

    comment_start_string: str = "{#"
    """String denoting the beginning of a comment (default: '{#')."""

    comment_end_string: str = "#}"
    """String denoting the end of a comment (default: '#}')."""

    line_statement_prefix: str | None = None
    """Prefix that begins a line-based statement (e.g., '#' for line statements)."""

    line_comment_prefix: str | None = None
    """Prefix that begins a line-based comment."""

    trim_blocks: bool = False
    """Remove first newline after a block (affects whitespace control)."""

    lstrip_blocks: bool = False
    """Remove leading spaces and tabs from the start of a line to a block."""

    newline_sequence: Literal["\n", "\r\n", "\r"] = "\n"
    """Sequence that starts a newline (default: '\n')."""

    keep_trailing_newline: bool = False
    """Preserve the trailing newline when rendering templates."""

    extensions: list[str] = Field(default_factory=list)
    """List of Jinja2 extensions to load (e.g., 'jinja2.ext.do')."""

    undefined: Literal["default", "strict", "debug", "chainable"] = "default"
    """Behavior when accessing undefined variables (default, strict, debug, chainable)."""

    filters: dict[str, str] = Field(default_factory=dict)
    """Custom filters as mapping of names to import paths."""

    tests: dict[str, str] = Field(default_factory=dict)
    """Custom tests as mapping of names to import paths."""

    globals: dict[str, Any] = Field(default_factory=dict)
    """Global variables available to all templates."""

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
            filters = {name: import_callable(path) for name, path in self.filters.items()}
            kwargs["filters"] = filters
            tests = {name: import_callable(path) for name, path in self.tests.items()}
            kwargs["tests"] = tests
        except Exception as exc:
            msg = f"Failed to import Jinja2 filters/tests: {exc}"
            raise ValueError(msg) from exc

        return kwargs


class GlobalSettings(ConfigModel):
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
    """Log level for LLMling core."""

    jinja_environment: Jinja2Config = Field(default_factory=Jinja2Config)
    """Jinja2 environment configuration"""

    llm_capabilities: LLMCapabilitiesConfig = Field(default_factory=LLMCapabilitiesConfig)
    """Control which system capabilities are exposed to LLMs."""


class BaseResource(Schema):
    """Base class for all resource types."""

    type: str = Field(init=False)
    """Type identifier for this resource."""

    description: str = ""
    """Human-readable description of the resource."""

    uri: str | None = None
    """Canonical URI for this resource, set during registration if unset."""

    processors: list[ProcessingStep] = Field(default_factory=list)
    """Processing steps to apply when loading this resource."""

    watch: WatchConfig | None = None
    """Configuration for file system watching, if supported."""

    name: str | None = Field(None, exclude=True)
    """Technical identifier (automatically set from config key during registration)."""

    # TODO: proper extra="forbid" for all subclasses.
    model_config = ConfigDict(frozen=True)

    def validate_resource(self) -> list[str]:
        """Validate resource at runtime.

        This validation is intentionally separated from Pydantic's model validation
        to handle:
        - Remote resources and URLs
        - Template/placeholder paths that don't exist during config loading
        - Resources that may become available after config loading
        - Validation of actual resource availability separate from config structure

        Returns:
            List of validation warnings (empty if all valid)
        """
        return []

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

    def get_watch_path(self) -> str | None:
        """Get the path to watch if this resource supports watching."""
        return None

    @property
    def mime_type(self) -> str:
        """Get the MIME type for this resource.

        This should be overridden by subclasses that can determine
        their MIME type. Default is text/plain.
        """
        return "text/plain"


class PathResource(BaseResource):
    """Resource loaded from a file or URL."""

    type: Literal["path"] = Field(default="path", init=False)
    """Discriminator field identifying this as a path-based resource."""

    path: str | os.PathLike[str]
    """Path to the file or URL to load."""

    watch: WatchConfig | None = None
    """Configuration for watching the file for changes."""

    def validate_resource(self) -> list[str]:
        """Check if path exists for local files."""
        from upathtools import to_upath

        path = to_upath(self.path)
        prefixes = ("http://", "https://")
        return [
            f"Resource path not found: {path}"
            for path in [path]
            if not path.exists() and not path.as_uri().startswith(prefixes)
        ]

    @property
    def supports_watching(self) -> bool:
        """Whether this resource instance supports watching."""
        from upathtools import to_upath

        path = to_upath(self.path)
        if not path.exists():
            msg = f"Cannot watch non-existent path: {self.path}"
            warnings.warn(msg, UserWarning, stacklevel=2)
            return False
        return True

    def get_watch_path(self) -> str | None:
        """Get the path to watch."""
        return str(self.path) if self.supports_watching else None

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

    type: Literal["text"] = Field(default="text", init=False)
    """Discriminator field identifying this as a text-based resource."""

    content: str
    """The actual text content of the resource."""

    _mime_type: str | None = None  # Optional override

    @model_validator(mode="after")
    def validate_content(self) -> TextResource:
        """Validate that the content is not empty."""
        if not self.content:
            msg = "Content cannot be empty"
            raise ValueError(msg)
        return self

    @property
    def mime_type(self) -> str:
        """Get MIME type, trying to detect JSON/YAML."""
        if self._mime_type:
            return self._mime_type
        # Could add content inspection here
        return "text/plain"


class CLIResource(BaseResource):
    """Resource from CLI command execution."""

    type: Literal["cli"] = Field(default="cli", init=False)
    """Discriminator field identifying this as a CLI-based resource."""

    command: str | TypingSequence[str]
    """Command to execute (string or sequence of arguments)."""

    shell: bool = False
    """Whether to run the command through a shell."""

    cwd: str | None = None
    """Working directory for command execution."""

    timeout: float | None = None
    """Maximum time in seconds to wait for command completion."""

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


class RepositoryResource(BaseResource):
    """Git repository content."""

    type: Literal["repository"] = Field("repository", init=False)
    repo_url: str
    """URL of the git repository."""

    ref: str = "main"
    """Git reference (branch, tag, or commit)."""

    path: str = ""
    """Path within the repository."""

    sparse_checkout: list[str] | None = None
    """Optional list of paths for sparse checkout."""

    user: str | None = None
    """Optional user name for authentication."""

    password: SecretStr | None = None
    """Optional password for authentication."""

    def validate_resource(self) -> list[str]:
        return [
            f"Repository {self.repo_url} has user but no password"
            for _ in [None]
            if self.user and not self.password
        ]


class SourceResource(BaseResource):
    """Resource from Python source code."""

    type: Literal["source"] = Field(default="source", init=False)
    """Discriminator field identifying this as a source code resource."""

    import_path: str
    """Dotted import path to the Python module or object."""

    recursive: bool = False
    """Whether to include submodules recursively."""

    include_tests: bool = False
    """Whether to include test files and directories."""

    @model_validator(mode="after")
    def validate_import_path(self) -> SourceResource:
        """Validate that the import path is properly formatted."""
        if not all(part.isidentifier() for part in self.import_path.split(".")):
            msg = f"Invalid import path: {self.import_path}"
            raise ValueError(msg)
        return self


class CallableResource(BaseResource):
    """Resource from executing a Python callable."""

    type: Literal["callable"] = Field(default="callable", init=False)
    """Discriminator field identifying this as a callable-based resource."""

    import_path: str
    """Dotted import path to the callable to execute."""

    keyword_args: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the callable."""

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


Resource = Annotated[
    PathResource | TextResource | CLIResource | SourceResource | CallableResource,
    Field(discriminator="type"),
]


class WatchConfig(ConfigModel):
    """Watch configuration for resources."""

    enabled: bool = False
    """Whether the watch is enabled"""

    patterns: list[str] | None = None
    """List of pathspec patterns (.gitignore style)"""

    ignore_file: str | None = None
    """Path to .gitignore-style file"""


class ToolHints(ConfigModel):
    """Configuration for tool execution hints."""

    read_only: bool | None = None
    """Hints that this tool only reads data without modifying anything"""

    destructive: bool | None = None
    """Hints that this tool performs destructive operations that cannot be undone"""

    idempotent: bool | None = None
    """Hints that this tool has idempotent behaviour"""

    open_world: bool | None = None
    """Hints that this tool can access / interact with external resources beyond the
    current system"""


class ToolConfig(ConfigModel):
    """Configuration for a tool."""

    import_path: str
    """Import path to the tool implementation (e.g. 'mymodule.tools.MyTool')"""

    name: str | None = None
    """Optional override for the tool's display name"""

    description: str | None = None
    """Optional override for the tool's description"""

    hints: ToolHints = ToolHints()
    """Hints about the tool's behavior and execution characteristics"""


class BaseToolsetConfig(ConfigModel):
    """Base configuration for toolsets."""


class OpenAPIToolsetConfig(BaseToolsetConfig):
    """Configuration for OpenAPI toolsets."""

    type: Literal["openapi"] = Field("openapi", init=False)
    """Discriminator field identifying this as an OpenAPI toolset."""

    spec: str = Field(...)
    """URL or path to the OpenAPI specification document."""

    base_url: str | None = Field(default=None)
    """Optional base URL for API requests, overrides the one in spec."""


class EntryPointToolsetConfig(BaseToolsetConfig):
    """Configuration for entry point toolsets."""

    type: Literal["entry_points"] = Field("entry_points", init=False)
    """Discriminator field identifying this as an entry point toolset."""

    module: str = Field(..., description="Python module path")
    """Python module path to load tools from via entry points."""


class CustomToolsetConfig(BaseToolsetConfig):
    """Configuration for custom toolsets."""

    type: Literal["custom"] = Field("custom", init=False)
    """Discriminator field identifying this as a custom toolset."""

    import_path: str = Field(...)
    """Dotted import path to the custom toolset implementation class."""

    @field_validator("import_path", mode="after")
    @classmethod
    def validate_import_path(cls, v: str) -> str:
        from llmling.tools.toolsets import ToolSet

        # v is already confirmed to be a str here
        try:
            cls = import_class(v)
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


class Config(ConfigModel):
    """Root configuration model."""

    version: str = "1.0"
    """Version string for this configuration format."""

    global_settings: GlobalSettings = Field(default_factory=GlobalSettings)
    """Global settings that apply to all components."""

    context_processors: dict[str, ProcessorConfig] = Field(default_factory=dict)
    """Content processors available for resource transformation."""

    resources: dict[str, Resource] = Field(default_factory=dict)
    """Resource definitions keyed by name."""

    resource_groups: dict[str, list[str]] = Field(default_factory=dict)
    """Groups of resources for logical organization."""

    tools: dict[str, ToolConfig] = Field(default_factory=dict)
    """Tool definitions keyed by name."""

    toolsets: dict[str, ToolsetConfig] = Field(default_factory=dict)
    """Toolset configurations for extensible tool collections."""

    prompts: dict[str, PromptType] = Field(default_factory=dict)
    """Prompt definitions keyed by name."""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="allow",  # extra fields used by the server for example.
    )

    # @model_validator(mode="before")
    # @classmethod
    # def populate_prompt_names(cls, data: dict[str, Any]) -> dict[str, Any]:
    #     """Populate prompt names from dictionary keys before validation."""
    #     if isinstance(data, dict) and "prompts" in data:
    #         prompts = data["prompts"]
    #         if isinstance(prompts, dict):
    #             # Add name to each prompt's data
    #             data["prompts"] = {
    #                 key: {
    #                     "name": key,
    #                     **(val if isinstance(val, dict) else val.model_dump()),
    #                 }
    #                 for key, val in prompts.items()
    #             }
    #     return data

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

    @classmethod
    def from_file(cls, path: JoinablePathLike) -> Self:
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
        import yamling

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
            logger.debug("Loaded raw configuration:")
            msg = "version=%s, resources=%d, tools=%d, toolsets=%d, prompts=%d"
            logger.debug(
                msg,
                config.version,
                len(config.resources),
                len(config.tools),
                len(config.toolsets),
                len(config.prompts),
            )
            return config


if __name__ == "__main__":
    from llmling import Config

    config = Config.from_file(config_resources.TEST_CONFIG)  # type: ignore[has-type]
    print(config)
