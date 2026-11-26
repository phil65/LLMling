# AGENTS.md

## Overview

LLMling is a YAML-driven backend for LLM applications: a config file declares
resources (content providers), prompts (message templates), tools (Python callables,
OpenAPI specs, entry-point toolsets) and processors. `RuntimeConfig` turns that config
into live registries that consumers query. It was the backend for `mcp-server-llmling`
and `llmling-agent` (the latter is now succeeded by agentpool). Python >= 3.14, pydantic/schemez based.

## Architecture

Control flow: YAML -> `Config` (pydantic) -> `RuntimeConfig` (registries) -> consumer.

- `src/llmling/config/models.py`: `Config` root model and all config models
  (`*Resource`, `ToolConfig`, `*ToolsetConfig`, `GlobalSettings`, `Jinja2Config`).
  Frozen models; discriminated by `type`.
- `src/llmling/config/runtime.py`: `RuntimeConfig`, the main entry point
  (`RuntimeConfig.open(path)` async ctx manager, `open_sync`). Owns the resource,
  loader, processor, prompt and tool registries; register/load/render/execute API.
- `src/llmling/config/utils.py`: `prepare_runtime()` builds the registries from a `Config`.
- `src/llmling/config/manager.py`: `ConfigManager` (load/validate/save a config).
- `src/llmling/config/store.py`: `ConfigStore`, named configs persisted in the platformdirs
  user config dir (used by the CLI `config` commands).
- `src/llmling/config/llm_tools.py`: `LLMTools`, tools that let an LLM inspect/extend the runtime.
- `src/llmling/core/baseregistry.py`: `BaseRegistry`, a `MutableMapping` base for all registries.
- `src/llmling/core/chain.py`: `ChainTool` (pipeline of tool calls).
- `src/llmling/resources/`: `ResourceLoader` base (`base.py`), `ResourceRegistry`
  (`registry.py`), loaders per resource type in `loaders/` (path, text, cli, source,
  callable, repository), file watching in `watching/`.
- `src/llmling/prompts/`: `StaticPrompt`, `DynamicPrompt`, `FilePrompt` (`models.py`),
  `PromptRegistry` with argument completions (`registry.py`), function introspection (`utils.py`).
  Converts to fastmcp prompts (`to_fastmcp_prompt`).
- `src/llmling/tools/`: `LLMCallableTool` / `BaseTool` (`base.py`), `ToolRegistry`,
  `ToolSet` base (`toolsets.py`), `OpenAPITools`, `EntryPointTools`.
- `src/llmling/processors/`: `BaseProcessor`, function `Processor`, `Jinja2Processor`, `ProcessorRegistry`.
- `src/llmling/cli/` + `src/llmling/__main__.py`: Typer CLI `llmling` (config, resource,
  tool, prompt; `server`/`agent` subcommands only if those packages import).
- `src/llmling/config_resources/`: bundled example YAML configs (`test.yml` is used by tests).
- `src/llmling/testing/`: test tools/processors, also exposed as the `llmling` entry point.

## Where to look

| Task | Location |
|---|---|
| New config field / resource type | `config/models.py` (+ loader in `resources/loaders/`, register in `resources/loaders/registry.py`) |
| Runtime API behaviour | `config/runtime.py` |
| Tool creation from callables | `tools/base.py` |
| OpenAPI toolsets | `tools/openapi.py` |
| Prompt formatting / completions | `prompts/models.py`, `prompts/registry.py` |
| CLI commands | `cli/*.py`, `__main__.py` |
| Config JSON schema | `scripts/generate_schema.py` -> `schema/config-schema.json` |
| Tests | `tests/` (`tests/resources/` for loaders), fixtures in `tests/conftest.py` |

## Commands

```bash
uv sync --all-extras                 # setup (CI: --no-group docs --no-group lint)
uv run pytest                        # tests (addopts: -m "not slow")
uv run ruff check . && uv run ruff format --check .
uv run ty check                      # type check (src only); warnings fail
uv run deptry src                    # dependency check
uv run mknodes build          # docs (duty: build / serve)
uv run duty lint | lint_check | test | update | generate_schema | check_schema
```

Pre-commit runs ruff, tombi, ty and the full pytest suite on every commit.

## Conventions

- No `from __future__ import annotations` (PEP 649; banned via ruff TID251); Google docstrings; line length 100;
  ruff with preview rules (also formats Python blocks in Markdown).
- Models subclass `schemez.Schema` / `ConfigModel` with attribute docstrings.
- Use `# ty: ignore[rule]` for checker limitations, never blanket ignores.
- Tests: pytest-asyncio in auto mode; markers `slow`, `unit`, `integration`.
- Paths go through `upathtools.to_upath` / `UPath` (fsspec URLs are supported).

## Pitfalls

- Copier-managed (template `copier-phil65`): `.github/workflows/`, `.pre-commit-config.yaml`,
  `duties.py`, `mkdocs.yml`, most of `pyproject.toml`. Re-run `copier update` instead of
  hand-editing; dependencies go below the `# Only add below (Copier)` markers.
- Optional integrations imported lazily: crewai, langchain_core, autogen_core (tool
  conversion), tiktoken (`tools/code.py`), GitPython (`git` extra), watchfiles. They are
  listed in `[tool.ty.analysis]` / `[tool.deptry]`.
- Extras `agent`/`server` install `llmling-agent` / `mcp-server-llmling`, which currently fail
  to import against current dependencies; `__main__.py` catches the ImportError.
- `duties.py:serve_mcp` calls `python -m llmling.server`, which does not exist.
- `test_repository_loader` creates real git repos and is skipped when `CI=true`; it strips
  `GIT_*` env vars so it does not touch the outer repo when run from a git hook.
- `uv.lock` is gitignored.
