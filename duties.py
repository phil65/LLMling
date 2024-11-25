from __future__ import annotations

from duty import duty


@duty(capture=False)
def build(ctx, *args: str):
    """Build a MkNodes page."""
    args_str = " " + " ".join(args) if args else ""
    ctx.run(f"uv run mknodes build{args_str}")


@duty(capture=False)
def serve(ctx, *args: str):
    """Serve a MkNodes page."""
    args_str = " " + " ".join(args) if args else ""
    ctx.run(f"uv run mknodes serve{args_str}")


@duty(capture=False)
def test(ctx, *args: str):
    """Serve a MkNodes page."""
    args_str = " " + " ".join(args) if args else ""
    ctx.run(f"uv run pytest{args_str}")


@duty(capture=False)
def clean(ctx):
    """Clean all files from the Git directory except checked-in files."""
    ctx.run("git clean -dfX")


@duty(capture=False)
def update(ctx):
    """Update all environment packages using pip directly."""
    ctx.run("uv lock --upgrade")
    ctx.run("uv sync --all-extras")


@duty(capture=False)
def lint(ctx):
    """Lint the code and fix issues if possible."""
    ctx.run("uv run ruff check --fix --unsafe-fixes .")
    ctx.run("uv run ruff format .")
    ctx.run("uv run mypy src/llmling/")


@duty(capture=False)
def lint_check(ctx):
    """Lint the code."""
    ctx.run("uv run ruff check .")
    ctx.run("uv run ruff format --check .")
    ctx.run("uv run mypy src/llmling/")


@duty(capture=False)
def version(ctx, *args: str):
    """Bump package version."""
    args_str = " " + " ".join(args) if args else ""
    ctx.run(f"hatch version{args_str}")


@duty(capture=False)
def serve_mcp(ctx):
    """Start the MCP server for debugging."""
    ctx.run("uv run python -m llmling.server")


@duty(capture=False)
def inspect(ctx):
    """Start the MCP Inspector to debug our server."""
    # Use the uvx variant since we're using uv
    ctx.run("npx -y @modelcontextprotocol/inspector uvx --directory . run llmling")


@duty(capture=False)
def inspect_config(ctx, config: str = "src/llmling/config_resources/test.yml"):
    """Start the MCP Inspector with a specific config file."""
    ctx.run(
        f"npx -y @modelcontextprotocol/inspector uvx --directory . run llmling {config}"
    )


# @duty(capture=False)
# def serve_mcp(ctx, config: str = "src/llmling/config_resources/test.yml"):
#     """Start the LLMLing MCP server with specified config."""
#     ctx.run(f"uv run python -m llmling.server {config}")

# @duty(capture=False)
# def inspect(ctx, config: str = "src/llmling/config_resources/test.yml"):
#     """Start the MCP Inspector with our server."""
#     # Using uvx for Python package execution
#     cmd = (
#         "npx -y @modelcontextprotocol/inspector"
#         " uv"
#         " --directory ."
#         " run"
#         " python"
#         f" -m llmling.server {config}"
#     )
#     ctx.run(cmd)
