"""Backtrader-compatible facade backed by tradelearn's backtest engine."""

import importlib
from typing import Any


from tradelearn.engine import analyzers, feeds, observers, sizers, stores
from tradelearn.engine.analyzer import Analyzer
from tradelearn.engine.base import TimeFrame
from tradelearn.engine.cerebro import Cerebro, OptReturn
from tradelearn.engine.comminfo import CommInfoBase, CommissionInfo
from tradelearn.engine.indicators import Indicator
from tradelearn.engine.observers import Observer
from tradelearn.engine.signal import (
    SIGNAL_LONG,
    SIGNAL_LONG_ANY,
    SIGNAL_LONG_INV,
    SIGNAL_LONGEXIT,
    SIGNAL_LONGEXIT_ANY,
    SIGNAL_LONGEXIT_INV,
    SIGNAL_LONGSHORT,
    SIGNAL_NONE,
    SIGNAL_SHORT,
    SIGNAL_SHORT_ANY,
    SIGNAL_SHORT_INV,
    SIGNAL_SHORTEXIT,
    SIGNAL_SHORTEXIT_ANY,
    SIGNAL_SHORTEXIT_INV,
    Signal,
    SignalStrategy,
)
from tradelearn.engine.sizers import Sizer
from tradelearn.engine.strategy import (
    LineSeries,
    Order,
    Position,
    Strategy,
    Trade,
)


from tradelearn.backtest.clock import date2num, num2date

# Aliases for Backtrader compatibility
obs = observers


def __getattr__(name: str) -> Any:
    if name == "talib":
        return importlib.import_module("tradelearn.engine.talib")
    raise AttributeError(f"module 'tradelearn.engine' has no attribute {name!r}")

__all__ = [
    "Cerebro",
    "OptReturn",
    "CommInfoBase",
    "LineSeries",
    "Order",
    "Position",
    "Sizer",
    "Strategy",
    "Trade",
    "Indicator",
    "TimeFrame",
    "Analyzer",
    "feeds",
    "analyzers",
    "observers",
    "sizers",
    "stores",
    "Observer",
    "obs",
    "CommissionInfo",
    "Signal",
    "SignalStrategy",
    "SIGNAL_NONE",
    "SIGNAL_LONGSHORT",
    "SIGNAL_LONG",
    "SIGNAL_LONG_INV",
    "SIGNAL_LONG_ANY",
    "SIGNAL_SHORT",
    "SIGNAL_SHORT_INV",
    "SIGNAL_SHORT_ANY",
    "SIGNAL_LONGEXIT",
    "SIGNAL_LONGEXIT_INV",
    "SIGNAL_LONGEXIT_ANY",
    "SIGNAL_SHORTEXIT",
    "SIGNAL_SHORTEXIT_INV",
    "SIGNAL_SHORTEXIT_ANY",
    "num2date",
    "date2num",
]
