from __future__ import annotations

import logging
from logging import Logger

from rich.logging import RichHandler


def configure_logging(level: int = logging.INFO) -> Logger:
    handler = RichHandler(rich_tracebacks=True, markup=True, show_path=False)
    handler.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    package_logger = logging.getLogger("annealing")
    package_logger.setLevel(level)
    return package_logger
