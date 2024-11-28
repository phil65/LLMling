from __future__ import annotations

import asyncio
import logging
import sys

import logfire

from llmling import config_resources
from llmling.server import serve


def configure_logging(enable_logfire: bool = True) -> None:
    """Configure logging with optional Logfire."""
    # Configure all logging to go to stderr
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,  # Explicitly use stderr
    )

    if enable_logfire:
        logfire.configure()


async def main() -> None:
    """Run the LLMling server."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else config_resources.TEST_CONFIG

    configure_logging(enable_logfire=True)  # Enable for CLI usage

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        await serve(config_path)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as exc:  # noqa: BLE001
        print(f"Fatal server error: {exc}", file=sys.stderr)
        sys.exit(1)


def run() -> None:
    """Entry point for the server."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
