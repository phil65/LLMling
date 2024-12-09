# LLMling CLI Guide

LLMling provides a command-line interface for managing resources, tools, and prompts. This guide explains how to use the CLI effectively.

> **Note**: Server-related commands are documented in the `mcp-server-llmling` package.

## Configuration Management

Before using LLMling, you need to set up a configuration file. The CLI provides several ways to manage configurations:

```bash
# Create a new config with basic settings
llmling config init myconfig.yml

# Register configs for easy access
llmling config add dev ~/configs/dev.yml
llmling config add prod s3://bucket/prod.yml   # Supports various protocols

# Set active config
llmling config set dev

# List registered configs
llmling config list
```

Once a config is set as active, you can run commands without specifying the config file. You can always override the active config using the `-c/--config` option:
```bash
llmling -c prod.yml resource list  # Use specific config
llmling resource list             # Use active config
```

## Working with Resources

Resources are the main input for LLMling. You can list, inspect, and load resources defined in your config:

```bash
# List all resources
llmling resource list

# Show details of a resource
llmling resource show my_resource

# Load resource content
llmling resource load my_resource
```

> **Tip**: Use `-o yaml` or `-o json` to get machine-readable output:
> ```bash
> llmling resource list -o json
> ```

## Using Tools

Tools are functions that can be called by LLMs. The CLI lets you explore and test tools:

```bash
# List available tools
llmling tool list

# Show tool documentation
llmling tool show my_tool

# Call a tool with arguments
llmling tool call open_url url=https://github.com
```

## Managing Prompts

Prompts define how LLMs interact with resources and tools:

```bash
# List available prompts
llmling prompt list

# Show prompt details
llmling prompt show my_prompt
```

## Global Options

All commands support these options:
| Option | Description |
|--------|-------------|
| `-c, --config` | Config file or name |
| `-o, --output-format` | Output format (text/json/yaml) |
| `-v, --verbose` | Enable debug logging |

## Shell Completion

LLMling provides shell completion for:
- Command names and options
- Registered config names
- Output formats

To enable completion:
```bash
# Install completion for your shell
llmling --install-completion bash  # or zsh/fish/powershell
```

## Common Workflows

1. **Initial Setup**:
   ```bash
   # Create and register a config
   llmling config init myconfig.yml
   llmling config add dev myconfig.yml
   llmling config set dev
   ```

2. **Resource Management**:
   ```bash
   # Check available resources
   llmling resource list

   # Inspect specific resources
   llmling resource show interesting_resource
   llmling resource load interesting_resource
   ```

3. **Tool Testing**:
   ```bash
   # Find available tools
   llmling tool list

   # Test a tool
   llmling tool show my_tool
   llmling tool call my_tool arg1=value1
   ```

## Tips

- Use `--help` with any command to see available options
- The active config is shown in help text and command output
- Most commands support different output formats for scripting
- Error messages include suggestions for fixing issues

## Error Handling

Common error scenarios and solutions:
- No config set: Use `-c` or set an active config
- Resource not found: Check `resource list` for available resources
- Tool errors: Check arguments with `tool show`
