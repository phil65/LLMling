"""Server entry point."""

from __future__ import annotations

import asyncio
import sys

from llmling.core.log import get_logger
from llmling.server.factory import create_runtime_config, create_server


logger = get_logger(__name__)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def main() -> None:
    """Run the LLMling server."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        runtime = create_runtime_config(config_path)
        async with runtime as r:
            server = create_server(r)
            await server.start(raise_exceptions=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception:
        logger.exception("Fatal server error")
        sys.exit(1)


def run() -> None:
    """Entry point for the server."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
