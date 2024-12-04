# Prompts

Prompts in LLMling are message templates that can be formatted with arguments. There are three types of prompts:
- Static prompts defined in YAML
- Function-based prompts from Python code
- File-based prompts from external files

## Static Prompts

Static prompts are defined directly in the YAML configuration:

```yaml
prompts:
  code_review:
    messages:
      # System message setting the role
      - role: system
        content: |
          You are a code reviewer specialized in Python.
          Focus on these aspects:
          - Code style (PEP 8)
          - Best practices
          - Performance

      # User message with template variables
      - role: user
        content: |
          Review this Python code:
          ```python
          {code}
          ```
          Focus areas: {focus_areas}

    # Argument definitions
    arguments:
      - name: code
        description: "Python code to review"
        required: true
        type_hint: str

      - name: focus_areas
        description: "Areas to focus on (one of: style, security, performance)"
        required: false
        default: "style"
        type_hint: "Literal['style', 'security', 'performance']"
```

### Message Content Types

Messages can contain different types of content:

```yaml
prompts:
  mixed_content:
    messages:
      # Simple text content
      - role: system
        content: "You are an assistant."

      # Resource reference
      - role: user
        content:
          type: resource
          content: "file:///data.txt"
          alt_text: "Input data"

      # Image content
      - role: user
        content:
          type: image_url
          content: "https://example.com/diagram.png"
          alt_text: "System diagram"

      # Multiple content items
      - role: user
        content:
          - type: text
            content: "Analyze this image:"
          - type: image_url
            content: "https://example.com/chart.png"
            alt_text: "Performance chart"
```

## Function-Based Prompts

Function-based prompts are created from Python functions:

```yaml
prompts:
  analyze_code:
    import_path: myapp.prompts.code_analysis
    description: "Analyze Python code structure"
    # Optional message template override
    template: |
      Analysis results for {code}:
      {result}
```

The corresponding Python function:

```python
from typing import Literal

AnalysisFocus = Literal["complexity", "typing", "security"]

def code_analysis(
    code: str,
    focus: AnalysisFocus = "complexity",
    include_metrics: bool = True
) -> dict[str, Any]:
    """Analyze Python code structure and metrics.

    Args:
        code: Python source code to analyze
        focus: Analysis focus area
        include_metrics: Whether to include numeric metrics

    Returns:
        Analysis results dictionary
    """
    # Function implementation...
```

Function prompts provide:
- Type checking through annotations
- Docstring-based documentation
- Argument validation
- Auto-completion support

## File-Based Prompts

File-based prompts load content from external files:

```yaml
prompts:
  documentation:
    type: file
    path: "./prompts/docs.md"
    format: markdown      # text, markdown, or jinja2
    watch: true          # Optional file watching
    arguments:
      - name: topic
        description: "Documentation topic"
        required: true
```

Example Markdown prompt file (`docs.md`):
```markdown
# Documentation Assistant

You are a technical documentation expert.

## Task
Write documentation for: {topic}

## Guidelines
- Use clear language
- Include examples
- Follow style guide
```

## Template Formatting

Prompts support several formatting features:

### Basic Variable Substitution
```yaml
content: "Hello, {name}!"
```

### Conditional Content
```yaml
content: |
  {greeting}
  {details if include_details else ''}
```

### Format Specifiers
```yaml
content: "Value: {number:.2f}"
```

### Resource Inclusion
```yaml
content: |
  System prompt: {system_prompt}
  Code to review: {code_file.content}
```

### Jinja2 Templates
When using the Jinja2 processor:
```yaml
content: |
  {% for item in items %}
  - {{ item.name }}: {{ item.value }}
  {% endfor %}
```

## Argument Validation

Prompts validate their arguments:

```yaml
arguments:
  - name: temperature
    description: "Temperature value"
    required: true
    type_hint: float  # Type checking
    # Validation through Pydantic
    validation:
      ge: 0.0
      le: 1.0

  - name: model
    description: "Model to use"
    type_hint: "Literal['gpt-3.5-turbo', 'gpt-4']"
    default: "gpt-3.5-turbo"
```

## Best Practices

- Use descriptive prompt names
- Document arguments clearly
- Provide sensible defaults
- Use type hints for validation
- Keep prompts focused and modular
- Consider using file-based prompts for long content
- Use resource references for dynamic content

## Next Steps

The next section covers [Tools](#tools), which can be used to extend the LLM's capabilities with custom functionality.
```
