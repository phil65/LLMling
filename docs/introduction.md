# LLMling Documentation

LLMling is a declarative framework for creating predictable LLM environments. It allows you to define resources, prompts, and tools in YAML configuration files, making LLM interactions more structured and maintainable.

## What is LLMling?

LLMling is a Python framework that provides a configuration-driven approach to LLM development. Instead of writing code to manage resources and prompts, you declare them in YAML files. This makes LLM applications:
- More maintainable through separation of configuration and code
- More predictable through structured resource management
- More reusable through templated prompts and modular tools

## Key Concepts & Philosophy

LLMling is built around providing three core components:

| Concept | Description |
|---------|-------------|
| Resources | Content providers that load and preprocess data (files, text, CLI output, etc.) |
| Prompts | Message templates that can be formatted with arguments |
| Tools | Python functions that can be called by the LLM |

The philosophy behind LLMling is:

> **Predictability through Declaration**
> By declaring the LLM's environment explicitly, we make its behavior more predictable and easier to maintain.

## Why Use LLMling?

- **Declarative Configuration**: Define your LLM's environment in YAML instead of code
- **Resource Management**: Structured handling of various content sources
- **Template System**: Reusable prompts with argument validation
- **Tool Integration**: Safe extension of LLM capabilities through Python functions
- **File Watching**: Automatic reload of resources when files change
- **Processing Pipeline**: Transform content before it reaches the LLM

## How Does LLMling Work?

1. You create a YAML configuration file defining:
   - Resources (files, text, commands, etc.)
   - Prompts (message templates)
   - Tools (Python functions)

2. LLMling loads this configuration and provides a runtime environment that:
   - Manages resource loading and caching
   - Handles prompt formatting and validation
   - Executes tools safely

## What Can You Do With LLMling?

```yaml
resources:
  python_files:
    type: path
    path: "./src/**/*.py"
    watch:
      enabled: true
      patterns: ["*.py"]

  review_prompt:
    type: text
    content: |
      You are a Python code reviewer.
      Review this code: {code}
      Focus on: {focus_areas}

tools:
  analyze:
    import_path: llmling.tools.code.analyze
```

```python
# Example: Create a code review environment

async with RuntimeConfig.from_file("path_to_config.yml") as runtime:
    # Load a resource
    code = await runtime.load_resource("python_files")

    # Format a prompt
    messages = await runtime.format_prompt(
        "review_prompt",
        code=code.content,
        focus_areas="style,security"
    )

    # Execute a tool
    analysis = await runtime.execute_tool("analyze", code=code.content)
```

## Getting Started

### Installation

```bash
pip install llmling
```

For development installations with all extras:
```bash
pip install "llmling[dev,test,docs]"
```

### Quick Start

1. Create a configuration file (`config.yml`):
```yaml
resources:
  greeting:
    type: text
    content: "Hello, {name}!"

prompts:
  welcome:
    messages:
      - role: system
        content: "You are a friendly assistant."
      - role: user
        content: "{greeting}"
```

2. Use the configuration:
```python
from llmling import RuntimeConfig

async with RuntimeConfig.from_file("config.yml") as runtime:
    greeting = await runtime.load_resource("greeting")
    formatted = greeting.format(name="World")
    print(formatted)  # "Hello, World!"
```

> **Note**
> LLMling requires Python 3.12 or later and uses modern Python features like the match statement and type hints.

For more detailed examples and use cases, see the [Example Configuration](#example-configuration) section.
```
