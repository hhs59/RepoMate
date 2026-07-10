from __future__ import annotations

import logging
import os

from rich.console import Console
from rich.logging import RichHandler


def setup_logging() -> None:
    level_name = os.environ.get("CIYC_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = RichHandler(console=Console(), rich_tracebacks=True, show_time=False)
    handler.setLevel(level)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
