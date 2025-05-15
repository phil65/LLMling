# YAML Configuration

LLMling uses YAML files for configuration. This section explains the main configuration sections and their purpose.

## Configuration Structure

A LLMling configuration file has this basic structure:

```yaml
global_settings:  # Global configuration
  requirements: []
  timeout: 30
  log_level: "INFO"

resources:        # Content sources
  resource_name:
    type: "path"
    # resource-specific config...

prompts:         # Message templates
  prompt_name:
    messages: []
    # prompt-specific config...

tools:           # Python functions
  tool_name:
    import_path: "module.path"
    # tool-specific config...

toolsets:        # Pre-built tool collections
  my_tools:
    type: entry_points
    module: llmling
```

## Global Settings

The `global_settings` section configures the runtime environment:

```yaml
global_settings:
  # Python package requirements
  requirements:
    - "package_name>=1.0.0"
    - "other-package"

  # Alternative PyPI index
  pip_index_url: "https://pypi.org/simple"

  # Additional import paths
  extra_paths:
    - "./src"
    - "./lib"

  # PEP 723 scripts (for remote code)
  scripts:
    - "https://gist.githubusercontent.com/user/script.py"

  # Package manager preference
  prefer_uv: true

  # Global timeout in seconds
  timeout: 30

  # Logging level
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

  # Jinja2 environment settings (for templates)
  jinja_environment:
    block_start_string: "{%"
    block_end_string: "%}"
    # ... other Jinja2 settings
```

## Resources

The `resources` section defines content sources. Resources can be files, text, command output, or other sources:

```yaml
resources:
  example_file:
    type: "path"           # Resource type
    path: "./data.txt"     # Resource-specific config
    description: "Optional description"
```

> **Note**
> Different resource types (`path`, `text`, `cli`, etc.) are covered in detail in the [Resource Types](#resource-types) section.

## Prompts

The `prompts` section defines message templates that can be formatted with arguments:

```yaml
prompts:
  example_prompt:
    messages:
      - role: "system"
        content: "System message"
      - role: "user"
        content: "User message with {argument}"
    arguments:
      - name: "argument"
        description: "Argument description"
        required: true
```

> **Note**
> For a complete overview of prompt features, see the [Prompts](#prompts) section.

## Tools

The `tools` section defines Python functions that can be called by the LLM:

```yaml
tools:
  example_tool:
    import_path: "package.module.function"
    description: "Tool description"
```

> **Note**
> For detailed information about tool types and features, see the [Tools](#tools) section.

## Toolsets

The `toolsets` section lets you define collections of related tools. There are three types of toolsets:

```yaml
toolsets:
  # Entry point toolsets (tools from Python packages)
  core_tools:
    type: entry_points
    module: llmling

  # OpenAPI toolsets (tools from API specs)
  petstore:
    type: openapi
    spec: "https://petstore.swagger.io/v2/swagger.json"
    base_url: "https://api.example.com"  # Optional API base URL

  # Custom toolsets (your own tool collections)
  custom:
    type: custom
    import_path: "myapp.tools.CustomToolSet"
```

### Entry Point Toolsets

Entry point toolsets load tools from Python packages that provide them through entry points:

```yaml
toolsets:
  llmling:
    type: entry_points
    module: llmling       # Package name
```

### OpenAPI Toolsets

OpenAPI toolsets automatically create tools from OpenAPI/Swagger specifications:

```yaml
toolsets:
  api:
    type: openapi
    spec: "https://api.example.com/openapi.json"  # URL or local path
    base_url: "https://api.example.com"           # Optional base URL
```

### Custom Toolsets

Custom toolsets load tool collections from your own Python classes:

```yaml
toolsets:
  custom:
    type: custom
    import_path: "myapp.tools.DatabaseTools"  # Your toolset class
```


## File Watching

Resources can be configured to watch for file changes:

```yaml
resources:
  watched_file:
    type: "path"
    path: "./config.yml"
    watch:
      enabled: true
      patterns:          # .gitignore style patterns
        - "*.yml"
        - "!.private/*"  # Exclude private files
      ignore_file: ".gitignore"  # Use existing ignore file
```

## Resource Processing

Resources can be processed through a pipeline of processors:

```yaml
# First define processors
context_processors:
  uppercase:
    import_path: "myapp.processors.to_upper"

# Then use them in resources
resources:
  processed:
    type: "text"
    content: "Hello"
    processors:
      - name: "uppercase"
        kwargs:           # Optional processor arguments
          extra: "value"
```

## Schema Validation

LLMling validates your configuration against a JSON schema. Common validation errors include:

- Missing required fields
- Invalid resource types
- Incorrect prompt message structure
- Invalid tool import paths

Error messages will point to the specific issue in your configuration.

## Next Steps

The following sections provide detailed information about:
- [Resource Types](#resource-types): Available resource types and their configuration
- [Prompts](#prompts): Prompt types and template features
- [Tools](#tools): Function-based and class-based tools
