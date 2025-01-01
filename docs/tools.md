# Tools

Tools in LLMling are Python functions or classes that can be called by the LLM. They provide a safe way to extend the LLM's capabilities with custom functionality.

## Function-Based Tools

Function-based tools are the simplest way to expose functionality to the LLM:

```yaml
tools:
  analyze_code:
    import_path: myapp.tools.code.analyze
    description: "Analyze Python code structure"  # Optional override

  format_python:
    import_path: myapp.tools.code.format_code
    name: "Format Python Code"  # Optional name override
```

The corresponding Python functions:

```python
from typing import Any

async def analyze_code(
    code: str,
    include_metrics: bool = True,
) -> dict[str, Any]:
    """Analyze Python code structure and complexity.

    Args:
        code: Python source code to analyze
        include_metrics: Whether to include numeric metrics

    Returns:
        Dictionary containing analysis results
    """
    # Function implementation...

def format_code(
    code: str,
    line_length: int = 88,
) -> str:
    """Format Python code using black.

    Args:
        code: Python code to format
        line_length: Maximum line length

    Returns:
        Formatted code
    """
    # Function implementation...
```

Function tools support:
- Both sync and async functions
- Type checking through annotations
- Docstring-based documentation
- Default arguments
- Return type validation

## Class-Based Tools

Class-based tools provide more control and can maintain state:

```yaml
tools:
  browser:
    import_path: llmling.tools.browser.BrowserTool
    description: "Browser automation tool"
```

The corresponding Python class:

```python
from typing import Literal
from llmling.tools.base import BaseTool

class BrowserTool(BaseTool):
    """Tool for web browser automation."""

    name = "browser"  # Tool name
    description = "Control web browser for research"  # Tool description

    async def startup(self) -> None:
        """Initialize browser on startup."""
        self.browser = await self._launch_browser()

    async def execute(
        self,
        action: Literal["open", "click", "read"],
        url: str | None = None,
        selector: str | None = None,
    ) -> dict[str, str]:
        """Execute browser action.

        Args:
            action: Browser action to perform
            url: URL to navigate to (for 'open' action)
            selector: Element selector (for 'click' and 'read' actions)

        Returns:
            Action result
        """
        match action:
            case "open":
                return await self._open_page(url)
            case "click":
                return await self._click_element(selector)
            case "read":
                return await self._read_content(selector)

    async def shutdown(self) -> None:
        """Clean up browser resources."""
        await self.browser.close()
```

Class-based tools provide:
- Resource lifecycle management (startup/shutdown)
- State maintenance
- Complex functionality encapsulation
- Type hints and validation
- Progress reporting

## Progress Reporting

Tools can report progress to clients:

```python
from llmling.tools.base import BaseTool

class AnalysisTool(BaseTool):
    name = "analyze"
    description = "Analyze code repository"

    async def execute(
        self,
        path: str,
        recursive: bool = True,
        _meta: dict[str, Any] | None = None,  # Progress tracking
    ) -> dict[str, Any]:
        """Analyze Python files in a directory.

        Args:
            path: Directory to analyze
            recursive: Whether to analyze subdirectories
            _meta: Progress tracking metadata
        """
        files = list(Path(path).glob("**/*.py" if recursive else "*.py"))

        for i, file in enumerate(files, 1):
            # Report progress if meta information provided
            if _meta and "progressToken" in _meta:
                self.notify_progress(
                    token=_meta["progressToken"],
                    progress=i,
                    total=len(files),
                    description=f"Analyzing {file.name}"
                )

            # Process file...
```

## Tool Collections (Toolsets)

Related tools can be grouped into toolsets:

```yaml
# In configuration:
toolsets:
    - llmling.code     # Code analysis tools
    - llmling.browser  # Browser automation
    - myapp.tools     # Custom tools
```

Creating a toolset:

```python
# myapp/tools.py
from typing import Callable, Any

def get_mcp_tools() -> list[Callable[..., Any]]:
    """Entry point exposing tools to LLMling."""
    from myapp.tools import analyze_code, format_code, check_style
    return [analyze_code, format_code, check_style]
```

Register the toolset in `pyproject.toml`:

```toml
[project.entry-points.llmling]
tools = "myapp.tools:get_mcp_tools"
```

## Best Practices

### Function Tools
- Use descriptive function names
- Provide clear docstrings
- Use type hints
- Handle errors gracefully
- Keep functions focused
- Use async for I/O operations

### Class Tools
- Inherit from `LLMCallableTool`
- Manage resources properly
- Report progress for long operations
- Validate input parameters
- Document behavior clearly
- Handle shutdown properly

### General
- Use meaningful error messages
- Follow Python naming conventions
- Keep tools simple and focused
- Test tools thoroughly
- Document requirements
- Use type hints consistently

## Next Steps

The next section covers [Extension Points](#extension-points), which explain how to extend LLMling's functionality.
