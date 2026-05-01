"""Lite public facade."""

from __future__ import annotations

import importlib
from typing import Any

from .backtest import Backtest
from .indicator import Signal
from .strategy import Strategy

SignalStrategy = Strategy

__all__ = [
    "Backtest",
    "MLStrategy",
    "Signal",
    "SignalStrategy",
    "Strategy",
    "ta",
    "talib",
    "tdx",
    "tv",
]


def __getattr__(name: str) -> Any:
    if name == "MLStrategy":
        from .ml_strategy import MLStrategy

        return MLStrategy
    if name == "ta":
        return importlib.import_module("tradelearn.indicators")
    if name == "talib":
        return importlib.import_module("tradelearn.indicators.talib")
    if name == "tdx":
        return importlib.import_module("tradelearn.indicators.tdx")
    if name == "tv":
        return importlib.import_module("tradelearn.indicators.tv")
    raise AttributeError(f"module 'tradelearn.lite' has no attribute {name!r}")
