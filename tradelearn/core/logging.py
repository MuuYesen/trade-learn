"""Logging helpers shared by trade-learn modules."""

from __future__ import annotations

import logging
import os

ENV_LOG_LEVEL = "TRADELEARN_LOG_LEVEL"


def configure_logging(level: str | int | None = None) -> None:
    """Configure standard Python logging for trade-learn."""

    selected = level if level is not None else os.environ.get(ENV_LOG_LEVEL, "INFO")
    logging.basicConfig(
        level=selected,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the tradelearn namespace."""

    if name.startswith("tradelearn"):
        return logging.getLogger(name)
    return logging.getLogger(f"tradelearn.{name}")
