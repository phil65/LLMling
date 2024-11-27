from __future__ import annotations

import sys


# just forward for now


def run():
    from llmling.server import __main__

    __main__.run()


if __name__ == "__main__":
    sys.exit(run())  # Make sure to handle exit codes properly
