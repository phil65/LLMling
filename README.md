# LLMling

[![PyPI License](https://img.shields.io/pypi/l/llmling.svg)](https://pypi.org/project/llmling/)
[![Package status](https://img.shields.io/pypi/status/llmling.svg)](https://pypi.org/project/llmling/)
[![Daily downloads](https://img.shields.io/pypi/dd/llmling.svg)](https://pypi.org/project/llmling/)
[![Weekly downloads](https://img.shields.io/pypi/dw/llmling.svg)](https://pypi.org/project/llmling/)
[![Monthly downloads](https://img.shields.io/pypi/dm/llmling.svg)](https://pypi.org/project/llmling/)
[![Distribution format](https://img.shields.io/pypi/format/llmling.svg)](https://pypi.org/project/llmling/)
[![Wheel availability](https://img.shields.io/pypi/wheel/llmling.svg)](https://pypi.org/project/llmling/)
[![Python version](https://img.shields.io/pypi/pyversions/llmling.svg)](https://pypi.org/project/llmling/)
[![Implementation](https://img.shields.io/pypi/implementation/llmling.svg)](https://pypi.org/project/llmling/)
[![Releases](https://img.shields.io/github/downloads/phil65/llmling/total.svg)](https://github.com/phil65/llmling/releases)
[![Github Contributors](https://img.shields.io/github/contributors/phil65/llmling)](https://github.com/phil65/llmling/graphs/contributors)
[![Github Discussions](https://img.shields.io/github/discussions/phil65/llmling)](https://github.com/phil65/llmling/discussions)
[![Github Forks](https://img.shields.io/github/forks/phil65/llmling)](https://github.com/phil65/llmling/forks)
[![Github Issues](https://img.shields.io/github/issues/phil65/llmling)](https://github.com/phil65/llmling/issues)
[![Github Issues](https://img.shields.io/github/issues-pr/phil65/llmling)](https://github.com/phil65/llmling/pulls)
[![Github Watchers](https://img.shields.io/github/watchers/phil65/llmling)](https://github.com/phil65/llmling/watchers)
[![Github Stars](https://img.shields.io/github/stars/phil65/llmling)](https://github.com/phil65/llmling/stars)
[![Github Repository size](https://img.shields.io/github/repo-size/phil65/llmling)](https://github.com/phil65/llmling)
[![Github last commit](https://img.shields.io/github/last-commit/phil65/llmling)](https://github.com/phil65/llmling/commits)
[![Github release date](https://img.shields.io/github/release-date/phil65/llmling)](https://github.com/phil65/llmling/releases)
[![Github language count](https://img.shields.io/github/languages/count/phil65/llmling)](https://github.com/phil65/llmling)
[![Github commits this week](https://img.shields.io/github/commit-activity/w/phil65/llmling)](https://github.com/phil65/llmling)
[![Github commits this month](https://img.shields.io/github/commit-activity/m/phil65/llmling)](https://github.com/phil65/llmling)
[![Github commits this year](https://img.shields.io/github/commit-activity/y/phil65/llmling)](https://github.com/phil65/llmling)
[![Package status](https://codecov.io/gh/phil65/llmling/branch/main/graph/badge.svg)](https://codecov.io/gh/phil65/llmling/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyUp](https://pyup.io/repos/github/phil65/llmling/shield.svg)](https://pyup.io/repos/github/phil65/llmling/)

[Read the documentation!](https://phil65.github.io/llmling/)


# LLMling User Manual

> [!WARNING]
> LLMling is under active development. APIs and configurations may change frequently. Check the repository for the latest updates.

LLMling is a flexible tool management system designed for LLM-based applications. It provides a modular approach to managing resources, processing tools, and prompt templates.

## Quick Start

Start the LLMling MCP server using [`uvx`](https://github.com/astral-sh/uv):

```bash
uvx --from llmling@latest mcp-server-llmling path/to/config.yml
```

## Core Concepts

### Resources

Resources are the basic building blocks in LLMling. They represent different types of content that can be loaded and processed.

> [!NOTE]
> Available resource types:
> - `text`: Raw text content
> - `path`: Files or URLs
> - `cli`: Command-line output
> - `source`: Python source code
> - `callable`: Python function results
> - `image`: Image files or URLs

### Resource Configuration

Resources are defined in YAML configuration files. Each resource needs a unique name and type:

```yaml
resources:
  guidelines:
    type: path  # A file resource from any origin
    path: "docs/guidelines.md"  # fsspec-backed, can also point to remote sources.
    description: "Coding guidelines"
    watch:  # Optional file watching
      enabled: true
      patterns: ["*.md"]

  system_info:  # The result of a python function call
    type: callable
    import_path: "platform.uname"
    description: "System information"

  code_sample:  # source code for module myapp.utils
    type: source
    import_path: "myapp.utils"
    recursive: true
    include_tests: false
```

> [!TIP]
> Add a `watch` section to automatically reload resources when files change. Use `.gitignore`-style patterns to control which files trigger updates.

### Prompts

LLMling supports two ways to define prompts: YAML configuration and Python functions.

#### YAML-Based Prompts

Define prompts directly in your configuration:

```yaml
prompts:
  code_review:
    description: "Review code changes"
    messages:
      - role: system
        content: |
          You are a code reviewer analyzing these changes:
          {changes}
    arguments:
      - name: changes
        type: text
        required: true
```

#### Function-Based Prompts

Convert Python functions into prompts automatically by leveraging type hints and docstrings:

```python
from typing import Literal

def analyze_code(
    code: str,
    language: str = "python",
    style: Literal["brief", "detailed"] = "brief",
    focus: list[str] | None = None,
) -> str:
    """Analyze code quality and structure.

    Args:
        code: Source code to analyze
        language: Programming language
        style: Analysis style (brief or detailed)
        focus: Optional areas to focus on (e.g. ["security", "performance"])
    """
    pass
```

Register in YAML:
```yaml
prompts:
  code_analyzer:
    import_path: "myapp.prompts.analyze_code"
    name: "Analyze Code"  # Optional override
    description: "Custom description"  # Optional override
    template: "Please analyze this {language} code:\n\n```{language}\n{code}\n```"  # Optional
```

The function will be automatically converted into a prompt with:
- Arguments derived from parameters
- Descriptions from docstrings
- Validation from type hints
- Enum values from Literal types
- Default values preserved

> [!TIP]
> Function-based prompts make it easy to create well-documented, type-safe prompts directly from your Python code. The automatic conversion handles argument validation, documentation, and template generation

## Custom Autocompletion

LLMling provides flexible autocompletion for prompt arguments, combining multiple sources of suggestions and allowing custom completion functions.

### Automatic Completions

Arguments automatically get completions based on:

1. Type hints:
```python
def analyze_code(
    language: Literal["python", "javascript", "rust"],
    style: bool = True,
    level: Literal["basic", "detailed"] | None = None,
) -> str:
    """Analyze code."""
    pass
```

2. Description hints:
```python
def analyze_text(
    text: str,
    format: str = "md"
) -> str:
    """Analyze text.

    Args:
        text: Input text
        format: Output format (one of: md, txt, rst)
    """
    pass
```

### Custom Completion Functions

You can provide custom completion functions either in code or via configuration:



### Via YAML Configuration:
```yaml
prompts:
  analyze_framework:
    import_path: "mymodule.analysis.analyze_framework"
    completions:
      framework: "mymodule.completions.get_framework_completions"
```

### Completion Priority

Completions are tried in this order:
1. Custom completion function (if provided)
2. Type-based completions (Literal, bool)
3. Description-based completions (from "one of:" syntax)
4. Default value (when no current input)

All completions are:
- Case-insensitive matched against current input
- Deduplicated while preserving order
- Limited to 100 results

### Writing Custom Completion Functions

Custom completion functions should:
1. Take a single string argument (current input)
2. Return a list of strings (possible completions)
3. Handle empty input appropriately
4. Be fast and handle errors gracefully

Example:
```python
def get_language_completions(current: str) -> list[str]:
    """Get programming language suggestions."""
    languages = ["python", "javascript", "typescript", "rust", "go"]

    # Handle empty input
    if not current:
        return languages

    # Filter based on current input
    current = current.lower()
    return [
        lang for lang in languages
        if lang.lower().startswith(current)
    ]
```

### Tools

Tools are Python functions or classes that can be called by LLMs. LLMling automatically generates OpenAI-compatible function schemas.

#### Function-Based Tools

The simplest way to create a tool is by using a regular Python function:

```python
async def analyze_code(code: str) -> dict[str, Any]:
    """Analyze Python code complexity and structure.

    Args:
        code: Python code to analyze

    Returns:
        Dictionary with analysis metrics
    """
    tree = ast.parse(code)
    return {
        "classes": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
        "functions": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
    }
```

Register in YAML:
```yaml
tools:
  code_analyzer:
    import_path: "myapp.tools.analyze_code"
    name: "Analyze Code"  # Optional override
    description: "Analyzes Python code structure"  # Optional override
```

> [!NOTE]
> LLMling automatically generates OpenAI function schemas from type hints and docstrings. No manual schema definition needed!


#### Class-Based Tools

For more complex tools, create a class inheriting from `LLMCallableTool`:

```python
class BrowserTool(LLMCallableTool):
    name = "browser"
    description = "Control a web browser"

    async def execute(
        self,
        action: Literal["open", "click", "read"],
        url: str | None = None,
        selector: str | None = None,
    ) -> dict[str, str]:
        """Execute browser actions."""
        match action:
            case "open":
                return await self._open_page(url)
            case "click":
                return await self._click_element(selector)
```

### Processors

Processors transform resource content before it's used. They can be chained together for complex transformations.

```yaml
resources:
  documentation:
    type: path
    path: "docs/"
    processors:
      - name: normalize_text
        parallel: false  # Run sequentially
        required: true
      - name: extract_sections
        parallel: true   # Can run in parallel
        kwargs:
          min_length: 100
```

#### Custom Processors

Create processors by implementing `BaseProcessor`:

```python
class TemplateProcessor(ChainableProcessor):
    async def _process_impl(self, context: ProcessingContext) -> ProcessorResult:
        template = self.config.template
        result = template.render(content=context.current_content)
        return ProcessorResult(
            content=result,
            original_content=context.original_content
        )
```

## Using with MCP Server

While LLMling's core functionality is independent, it includes an [MCP](https://github.com/microsoft/mcp) server implementation for remote tool execution:

```python
from llmling.server import serve

# Start MCP server with config
await serve("config.yml")
```

> [!TIP]
> The core LLMling functionality works without the MCP server. Use the components directly in your application or create custom server implementations.

## Advanced Features

### Dynamic Tool Registration

Register tools from Python code:

```python
from llmling.tools import ToolRegistry

registry = ToolRegistry()

# Register a module's public functions
registry.add_container(my_module, prefix="utils_")

# Register individual function
registry.register("analyze", analyze_function)
```

### Resource Groups

Group related resources for easier management:

```yaml
resource_groups:
  code_review:
    - python_files
    - lint_config
    - style_guide
```

### Extension System

Libraries can expose their functionality to LLMling without having to create a full MCP server implementation. This is done via entry points:

```toml
# In your library's pyproject.toml
[project.entry-points.llmling]
tools = "your_library:get_mcp_tools"  # Function returning list of callables
```

```python
# In your library
def get_mcp_tools() -> list[Callable[..., Any]]:
    """Expose functions as LLM tools."""
    return [
        analyze_code,
        validate_json,
        process_data,
    ]
```

Enable tools from a package in your LLMling configuration:

```yaml
# llmling.yml
toolsets:
  - your_library  # Use tools from your_library
```

> [!TIP]
> Libraries can expose their most useful functions as LLM tools without any LLMling-specific code. The entry point system uses Python's standard packaging features.

#### Discoverable Tools

Tools exposed through entry points:
- Are automatically discovered
- Get schemas generated from type hints and docstrings
- Can be used like any other LLMling tool
- Don't require the library to depend on LLMling

This allows for a rich ecosystem of tools that can be easily composed and used by LLMs.
