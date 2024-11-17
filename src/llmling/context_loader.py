"""Context loading utilities."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

from upath import UPath


if TYPE_CHECKING:
    import os


class ContextError(Exception):
    """Base exception for context-related errors."""


async def load_path_context(path: str | os.PathLike[str]) -> str:
    """Load context from file or URL."""
    upath = UPath(path)

    try:
        return upath.read_text()
    except Exception as exc:
        msg = f"Failed to load context from {path}"
        raise ContextError(msg) from exc


async def load_cli_context(
    command: str | list[str],
    shell: bool = False,
    cwd: str | None = None,
    timeout: int | None = None,
) -> str:
    """Execute CLI command and return output."""
    try:
        return subprocess.check_output(
            command,
            shell=shell,
            cwd=cwd,
            timeout=timeout,
            text=True,
        )
    except subprocess.SubprocessError as exc:
        msg = f"Failed to execute command: {command}"
        raise ContextError(msg) from exc


async def load_context(context_config: dict[str, Any]) -> str:
    """Load context based on its type."""
    context_type = context_config["type"]

    if context_type == "path":
        return await load_path_context(context_config["path"])

    if context_type == "text":
        return context_config["content"]

    if context_type == "cli":
        return await load_cli_context(
            command=context_config["command"],
            shell=context_config.get("shell", False),
            cwd=context_config.get("cwd"),
            timeout=context_config.get("timeout"),
        )

    msg = f"Unknown context type: {context_type}"
    raise ContextError(msg)


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        # Example with path context
        path_config = {"type": "path", "path": "example.txt"}
        try:
            path_content = await load_context(path_config)
            print("Path content:", path_content)
        except ContextError as e:
            print(f"Error loading path: {e}")

        # Example with text context
        text_config = {"type": "text", "content": "Hello from text context!"}
        text_content = await load_context(text_config)
        print("Text content:", text_content)

        # Example with CLI context
        cli_config = {"type": "cli", "command": "echo Hello from CLI!", "shell": True}
        try:
            cli_output = await load_context(cli_config)
            print("CLI output:", cli_output)
        except ContextError as e:
            print(f"Error executing command: {e}")

    asyncio.run(main())
