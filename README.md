# LLMling

[![PyPI License](https://img.shields.io/pypi/l/LLMling.svg)](https://pypi.org/project/LLMling/)
[![Package status](https://img.shields.io/pypi/status/LLMling.svg)](https://pypi.org/project/LLMling/)
[![Monthly downloads](https://img.shields.io/pypi/dm/LLMling.svg)](https://pypi.org/project/LLMling/)
[![Distribution format](https://img.shields.io/pypi/format/LLMling.svg)](https://pypi.org/project/LLMling/)
[![Wheel availability](https://img.shields.io/pypi/wheel/LLMling.svg)](https://pypi.org/project/LLMling/)
[![Python version](https://img.shields.io/pypi/pyversions/LLMling.svg)](https://pypi.org/project/LLMling/)
[![Implementation](https://img.shields.io/pypi/implementation/LLMling.svg)](https://pypi.org/project/LLMling/)
[![Releases](https://img.shields.io/github/downloads/phil65/LLMling/total.svg)](https://github.com/phil65/LLMling/releases)
[![Github Contributors](https://img.shields.io/github/contributors/phil65/LLMling)](https://github.com/phil65/LLMling/graphs/contributors)
[![Github Discussions](https://img.shields.io/github/discussions/phil65/LLMling)](https://github.com/phil65/LLMling/discussions)
[![Github Forks](https://img.shields.io/github/forks/phil65/LLMling)](https://github.com/phil65/LLMling/forks)
[![Github Issues](https://img.shields.io/github/issues/phil65/LLMling)](https://github.com/phil65/LLMling/issues)
[![Github Issues](https://img.shields.io/github/issues-pr/phil65/LLMling)](https://github.com/phil65/LLMling/pulls)
[![Github Watchers](https://img.shields.io/github/watchers/phil65/LLMling)](https://github.com/phil65/LLMling/watchers)
[![Github Stars](https://img.shields.io/github/stars/phil65/LLMling)](https://github.com/phil65/LLMling/stars)
[![Github Repository size](https://img.shields.io/github/repo-size/phil65/LLMling)](https://github.com/phil65/LLMling)
[![Github last commit](https://img.shields.io/github/last-commit/phil65/LLMling)](https://github.com/phil65/LLMling/commits)
[![Github release date](https://img.shields.io/github/release-date/phil65/LLMling)](https://github.com/phil65/LLMling/releases)
[![Github language count](https://img.shields.io/github/languages/count/phil65/LLMling)](https://github.com/phil65/LLMling)
[![Github commits this month](https://img.shields.io/github/commit-activity/m/phil65/LLMling)](https://github.com/phil65/LLMling)
[![Package status](https://codecov.io/gh/phil65/LLMling/branch/main/graph/badge.svg)](https://codecov.io/gh/phil65/LLMling/)
[![PyUp](https://pyup.io/repos/github/phil65/LLMling/shield.svg)](https://pyup.io/repos/github/phil65/LLMling/)

**Declarative resources, prompts and tools for LLM applications, defined in YAML.**

LLMling turns a YAML file into a runtime that serves content (resources), message
templates (prompts) and Python callables (tools) to an LLM consumer. It was built as the
backend for [mcp-server-llmling](https://github.com/phil65/mcp-server-llmling) and
[llmling-agent](https://github.com/phil65/llmling-agent); its ideas live on in
[agentpool](https://github.com/phil65/agentpool).

[Read the documentation!](https://phil65.github.io/LLMling/)

## Installation

```bash
pip install llmling
```

Optional extras: `llmling[git]` (repository resources), `llmling[watch_files]`
(file watching via watchfiles), `llmling[fsspec_extras]` (HTTP/GitHub filesystems).

## Quickstart

Define an environment in `config.yml`:

```yaml
resources:
  greeting:
    type: text
    content: "Hello from LLMling!"
  notes:
    type: path
    path: "./notes.md"
    description: "Project notes"

tools:
  word_count:
    import_path: "my_tools.word_count"  # any importable callable
    description: "Count words in a text"

prompts:
  summarize:
    type: text
    description: "Summarize a text"
    messages:
      - role: system
        content: "You are a concise assistant."
      - role: user
        content: "Summarize this:\n{text}"
    arguments:
      - name: text
        description: "Text to summarize"
        required: true
```

With `my_tools.py` on the import path:

```python
def word_count(text: str) -> int:
    """Count the words in a text."""
    return len(text.split())
```

Use it from Python:

```python
import asyncio

from llmling import RuntimeConfig


async def main() -> None:
    async with RuntimeConfig.open("config.yml") as runtime:
        print(runtime.list_resource_names())  # ['greeting', 'notes']

        resource = await runtime.load_resource("greeting")
        print(resource.content)  # Hello from LLMling!

        count = await runtime.execute_tool("word_count", text=resource.content)
        print(count)  # 3

        messages = await runtime.render_prompt("summarize", {"text": resource.content})
        for message in messages:
            print(message.role, message.get_text_content())


asyncio.run(main())
```

Or from the command line:

```bash
llmling config init my_config.yml         # create a starter config
llmling config add myconfig my_config.yml # store it under a name
llmling config set myconfig               # make it the active config

llmling resource list -c config.yml
llmling resource load greeting -c config.yml
llmling tool call word_count text="a b c" -c config.yml
llmling prompt show summarize -c config.yml
```

## Features

- **Resources**: `path` (files, directories, globs, any fsspec/UPath URL, optional file
  watching), `text`, `cli` (command output), `source` (Python module source),
  `callable` (return value of a Python function) and `repository` (files from a git repo).
  Resources can be grouped (`resource_groups`) and run through processors.
- **Prompts**: static message templates (`type: text`), function-based prompts
  (`type: function`, arguments derived from the signature and docstring) and file-based
  prompts (`type: file`), with argument validation and completions.
- **Tools**: any importable callable becomes a tool with an OpenAI-compatible schema.
  Toolsets add collections of tools from OpenAPI specs (`type: openapi`), entry points
  (`type: entry_points`) or custom `ToolSet` classes (`type: custom`).
- **Processors**: function-based and Jinja2 processors to transform resource content.
- **Runtime registration**: register resources, tools and prompts at runtime via
  `RuntimeConfig` (`register_tool`, `register_static_prompt`, ...).
- **Dependencies**: `global_settings.requirements` and PEP 723 `scripts` declare
  packages needed by tools and prompts.
- **Typed**: pydantic models throughout; a JSON schema for the config format lives in
  `schema/config-schema.json`.

## More resource and toolset examples

```yaml
global_settings:
  requirements: ["myapp"]  # installed on demand if missing

resources:
  python_files:
    type: path
    path: "./src/**/*.py"
    watch:
      enabled: true
      patterns: ["*.py", "!**/__pycache__/**"]
  git_changes:
    type: cli
    command: "git diff HEAD~1"
    shell: true
    timeout: 5.0
  utils_module:
    type: source
    import_path: myapp.utils
    recursive: true
  system_info:
    type: callable
    import_path: platform.uname

resource_groups:
  code_review: [python_files, git_changes]

toolsets:
  petstore:
    type: openapi
    spec: "https://petstore3.swagger.io/api/v3/openapi.json"
  my_plugins:
    type: entry_points
    module: myapp
```

## Further reading

The [docs/](https://github.com/phil65/LLMling/tree/main/docs) folder contains longer
guides on the [YAML configuration](https://github.com/phil65/LLMling/blob/main/docs/yaml_config.md),
[resources](https://github.com/phil65/LLMling/blob/main/docs/resources.md),
[prompts](https://github.com/phil65/LLMling/blob/main/docs/prompts.md),
[tools](https://github.com/phil65/LLMling/blob/main/docs/tools.md) and
[the CLI](https://github.com/phil65/LLMling/blob/main/docs/cli.md).
The API reference is on the [documentation site](https://phil65.github.io/LLMling/).
