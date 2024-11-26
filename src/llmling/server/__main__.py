"""Main entry point for the LLMling server."""

from __future__ import annotations

import asyncio
import logging
import sys

from llmling.server import create_server


if sys.platform == "win32":
    # Force WindowsSelectorEventLoopPolicy on Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def main() -> None:
    """Run the LLMling server."""
    config_path = (
        sys.argv[1] if len(sys.argv) > 1 else "src/llmling/config_resources/test.yml"
    )

    try:
        server = create_server(config_path)
        await server.start(raise_exceptions=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as exc:  # noqa: BLE001
        print(f"Fatal server error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
