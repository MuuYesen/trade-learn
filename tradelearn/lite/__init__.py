"""Lite public facade."""

from __future__ import annotations

import importlib
from typing import Any

from tradelearn.backtest.targets import TargetOrderConstraints

from .backtest import Backtest
from .strategy import Strategy

__all__ = [
    "Backtest",
    "Strategy",
    "TargetOrderConstraints",
]


def __getattr__(name: str) -> Any:
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
    raise AttributeError(f"module 'tradelearn.lite' has no attribute {name!r}")
