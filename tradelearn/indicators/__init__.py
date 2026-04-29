"""Generic technical indicator namespace.

The default ``ta.*`` functions are thin wrappers over pandas-ta-classic.
Same-name indicators may differ from future ``ta.tdx`` and ``ta.tv`` namespaces.
Choose the namespace for the market convention you need.
"""

from __future__ import annotations

import importlib
from typing import Any

from tradelearn.indicators.base import FunctionIndicator, Indicator
from tradelearn.indicators.core.momentum import rsi
from tradelearn.indicators.core.overlap import bbands, ema, sma
from tradelearn.indicators.core.trend import adx, macd
from tradelearn.indicators.core.volatility import atr
from tradelearn.indicators.core.volume import vwap

__all__ = [
    "FunctionIndicator",
    "Indicator",
    "adx",
    "atr",
    "bbands",
    "ema",
    "macd",
    "rsi",
    "sma",
    "tdx",
    "tdx30",
    "vwap",
]


def __getattr__(name: str) -> Any:
    """Lazily expose market-specific indicator namespaces."""
    if name in {"tdx", "tdx30"}:
        return importlib.import_module("tradelearn.indicators.tdx")
    if name == "tv":
        return importlib.import_module("tradelearn.indicators.tv")
    raise AttributeError(f"module 'tradelearn.indicators' has no attribute {name!r}")
