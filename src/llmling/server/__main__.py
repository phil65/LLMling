from __future__ import annotations

import asyncio
import sys
from typing import Any

from llmling.core.log import get_logger
from llmling.server.factory import create_runtime_config
from llmling.server.server import LLMLingServer


logger = get_logger(__name__)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def main() -> None:
    """Run the LLMling server."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    transport = sys.argv[2] if len(sys.argv) > 2 else "stdio"  # noqa: PLR2004
    transport_options: dict[str, Any] = {}
    assert transport in {"stdio", "sse"}
    # Parse transport options from command line
    if transport == "sse" and len(sys.argv) > 3:  # noqa: PLR2004
        transport_options["host"] = sys.argv[3]
        if len(sys.argv) > 4:  # noqa: PLR2004
            transport_options["port"] = int(sys.argv[4])

    try:
        runtime = create_runtime_config(config_path)
        async with runtime as r:
            server = LLMLingServer(
                r,
                transport=transport,  # type: ignore
                transport_options=transport_options,
            )
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
