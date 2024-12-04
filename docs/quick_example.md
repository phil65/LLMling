
# Example Configuration

This section provides a complete example configuration showcasing the main features of LLMling.

## Basic Configuration

Here's a complete example showing LLMling's key features:

```yaml
# config.yml

global_settings:
  # Dependencies needed for functions/tools
  requirements:
    - "python-ast-explorer"
    - "black"
  timeout: 30  # Global timeout in seconds
  log_level: "INFO"

# Content sources
resources:
  # File resources with watching
  python_files:
    type: path
    path: "./src/**/*.py"
    watch:
      enabled: true
      patterns: ["*.py", "!**/__pycache__/**"]
    processors:  # Process files before use
      - name: format_python
        kwargs: { line_length: 88 }

  # Static text content
  system_prompt:
    type: text
    content: |
      You are a Python code reviewer specialized in:
      - Code style (PEP 8)
      - Best practices
      - Type hints
      - Documentation

  # Command output
  git_changes:
    type: cli
    command: "git diff HEAD~1"
    shell: true
    processors:
      - name: strip_ansi  # Remove ANSI color codes

# Message templates
prompts:
  # Static prompt with arguments
  code_review:
    messages:
      - role: system
        content: "{system_prompt}"
      - role: user
        content: |
          Review this Python code:

          {code}

          Focus areas: {focus_areas}
    arguments:
      - name: code
        description: "Python code to review"
        required: true
      - name: focus_areas
        description: "Areas to focus on (one of: style, typing, security)"
        required: false
        default: "style"

  # Function-based prompt
  analyze_dependencies:
    import_path: myapp.prompts.analyze_imports
    description: "Analyze Python module dependencies"

# Python functions callable by LLM
tools:
  # Function-based tool
  analyze_code:
    import_path: llmling.tools.code.analyze
    description: "Analyze Python code structure"

  # Class-based tool
  formatter:
    import_path: llmling.tools.code.CodeFormatter
    description: "Format Python code using black"

# Include pre-built tools
toolsets:
  - llmling.code  # Code analysis tools
```

## Using the Configuration

Here's how to use this configuration programmatically:

```python
from llmling import RuntimeConfig

async with RuntimeConfig.from_file("config.yml") as runtime:
    # Load resource content
    python_files = await runtime.load_resource("python_files")
    git_diff = await runtime.load_resource("git_changes")

    # Format prompt with arguments
    messages = await runtime.format_prompt(
        "code_review",
        code=python_files.content,
        focus_areas="typing,security"
    )

    # Execute tools
    analysis = await runtime.execute_tool(
        "analyze_code",
        code=python_files.content
    )
```

## Configuration Features

The example above demonstrates several key features:

### Resource Management
- File watching with patterns
- Content processing pipeline
- Multiple resource types (files, text, CLI)

### Prompt Templates
- Static prompts with arguments
- Function-based prompts
- Argument validation

### Tool Integration
- Function-based tools
- Class-based tools
- Pre-built toolsets

### Global Settings
- Package dependencies
- Timeouts
- Logging configuration

## Validation

LLMling validates your configuration at load time:
- Resource existence and access
- Prompt argument requirements
- Tool availability and dependencies
