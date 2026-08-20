"""Lightweight logger setup for scripts and library code."""

from __future__ import annotations

import logging
import sys


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def get_logger(
    name: str = "tsl",
    *,
    level: int | str = logging.INFO,
    stream: bool = True,
) -> logging.Logger:
    """Return a configured logger.

    Handlers are attached only once per logger name so repeated calls are safe.
    """
    logger = logging.getLogger(name)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    if stream and not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
        logger.addHandler(handler)
        # Avoid double-logging through the root logger.
        logger.propagate = False

    return logger
