"""trade-learn public package namespace."""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.2.0"

__all__ = [
    "__version__",
    "optimize",
    "ta",
    "pta",
    "talib",
    "tdx",
    "tv",
]


def __getattr__(name: str) -> Any:
    """Lazily expose public namespace aliases."""
    if name == "ta":
        return importlib.import_module("tradelearn.indicators")
    if name == "pta":
        return importlib.import_module("tradelearn.indicators.pta")
    if name == "talib":
        return importlib.import_module("tradelearn.indicators.talib")
    if name == "tdx":
        return importlib.import_module("tradelearn.indicators.tdx")
    if name == "tv":
        return importlib.import_module("tradelearn.indicators.tv")
    if name == "optimize":
        return importlib.import_module("tradelearn.optimize")
    if name == "lab":
        return importlib.import_module("tradelearn.lab")
    if name == "cli":
        return importlib.import_module("tradelearn.cli")
    raise AttributeError(f"module 'tradelearn' has no attribute {name!r}")
