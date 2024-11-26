"""Main entry point for the LLMling server."""

from __future__ import annotations

import asyncio
import sys

from llmling.server import create_server


async def main() -> None:
    """Run the LLMling server."""
    # Use provided config path or default
    config_path = (
        sys.argv[1] if len(sys.argv) > 1 else "src/llmling/config_resources/test.yml"
    )

    try:
        # Create and start server
        server = create_server(config_path)
        async with server:
            await server.start()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as exc:  # noqa: BLE001
        print(f"Fatal server error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
