"""Convenience entrypoint for starting the OneLLM backend locally.

Running ``python main.py`` from this directory is equivalent to:

    python litellm/proxy/proxy_cli.py --config config.yaml --port 4000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

DEFAULT_CONFIG = "config.yaml"
DEFAULT_PORT = "4000"


def _has_option(args: Sequence[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def _has_config_option(args: Sequence[str]) -> bool:
    return _has_option(args, "--config") or any(
        arg == "-c" or arg.startswith("-c=") for arg in args
    )


def _with_default_startup_args(args: Sequence[str]) -> list[str]:
    startup_args = list(args)
    if not _has_config_option(startup_args):
        startup_args[:0] = ["--config", DEFAULT_CONFIG]
    if not _has_option(startup_args, "--port"):
        startup_args[:0] = ["--port", DEFAULT_PORT]
    return startup_args


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    from litellm.proxy.proxy_cli import run_server

    run_server.main(
        args=_with_default_startup_args(sys.argv[1:]),
        prog_name="python main.py",
    )


if __name__ == "__main__":
    main()
