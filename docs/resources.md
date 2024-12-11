# Resource Types

LLMling supports several resource types for different content sources. Each type has specific configuration options and capabilities.

## Path Resource

The `path` resource type loads content from files or URLs:

```yaml
resources:
  # Single file
  config_file:
    type: path
    path: "./config.yml"
    description: "Configuration file"
    watch:
      enabled: true     # Optional file watching
      patterns:         # .gitignore style patterns
        - "*.yml"
        - "!.private/*"

  # Directory with pattern
  python_files:
    type: path
    path: "./src/**/*.py"  # Glob pattern
    watch:
      enabled: true
      patterns: ["*.py"]
      ignore_file: ".gitignore"  # Use existing ignore file

  # Remote file
  readme:
    type: path
    path: "https://raw.githubusercontent.com/user/repo/main/README.md"
```

### Path Resource Features
- Supports local files and HTTP(S) URLs
- Directory traversal with glob patterns
- File change watching
- Binary file support (images, PDFs)
- URI templates for dynamic paths

## Text Resource

The `text` resource type provides static text content:

```yaml
resources:
  system_prompt:
    type: text
    content: |
      You are an assistant that:
      - Speaks professionally
      - Uses bullet points
      - Stays concise

  template_text:
    type: text
    content: "Hello, {name}!"  # Supports templating
    description: "Greeting template"
```

### Text Resource Features
- Multiline text with YAML block scalars
- Template support with variables
- No file system access needed
- Ideal for static prompts

## CLI Resource

The `cli` resource type executes commands and captures their output:

```yaml
resources:
  git_changes:
    type: cli
    command: "git diff HEAD~1"  # String command
    shell: true                 # Use shell
    cwd: "./src"               # Working directory
    timeout: 5.0               # Command timeout

  docker_status:
    type: cli
    command:                   # List of arguments
      - "docker"
      - "ps"
      - "--format"
      - "{{.Names}}"
    shell: false              # Direct execution
```

### CLI Resource Features
- Shell and direct command execution
- Working directory configuration
- Command timeouts
- Output processing
- Environment variable support

## Source Resource

The `source` resource type loads Python source code:

```yaml
resources:
  module_source:
    type: source
    import_path: "myapp.utils"
    recursive: true           # Include submodules
    include_tests: false      # Exclude test files
    description: "Utility module source"
```

### Source Resource Features
- Python module source access
- Recursive module traversal
- Test file filtering
- Source code formatting

## Callable Resource

The `callable` resource type executes Python functions:

```yaml
resources:
  system_info:
    type: callable
    import_path: "platform.uname"
    keyword_args:             # Function arguments
      aliased: true
    description: "System information"

  database_stats:
    type: callable
    import_path: "myapp.db.get_stats"
    keyword_args:
      include_details: true
```

### Callable Resource Features
- Python function execution
- Argument passing
- Async function support
- Result caching


## Resource Groups

Resources can be organized into groups for easier access:

```yaml
resource_groups:
  documentation:
    - readme
    - diagram
    - api_docs

  code_review:
    - python_files
    - git_changes
    - review_prompt
```

## Resource Processing

All resource types support processing pipelines:

```yaml
resources:
  processed_file:
    type: path
    path: "input.txt"
    processors:
      - name: uppercase           # Transform to uppercase
      - name: template            # Apply templating
        kwargs:
          variables:
            name: "User"
      - name: validate           # Optional validation
        required: false          # Skip on failure
```

## Best Practices

- Use descriptive resource names
- Provide descriptions for complex resources
- Use resource groups for related content
- Enable file watching only when needed
- Set appropriate timeouts for CLI/callable resources
- Use processors for consistent content formatting

## Next Steps

The next section covers [Prompts](#prompts), which can use these resources as content sources.
