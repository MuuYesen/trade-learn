"""Backtrader-compatible facade backed by tradelearn's backtest engine."""

import importlib
from typing import Any

import pandas as pd

from tradelearn.engine import analyzers, feeds, observers, sizers
from tradelearn.engine.analyzer import Analyzer
from tradelearn.engine.base import TimeFrame
from tradelearn.engine.cerebro import Cerebro, OptReturn
from tradelearn.engine.grid import GridSearchResult, grid_search
from tradelearn.engine.index_enhance import IndexEnhanceStrategy
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
from tradelearn.engine.sizers import AllInSizer, FixedSize, PercentSizer, Sizer
from tradelearn.engine.strategy import (
    CommInfoBase,
    DataFeed,
    ExecutedInfo,
    LineSeries,
    Order,
    Params,
    Position,
    Strategy,
    Trade,
)


def num2date(num):
    """Convert numeric timestamp back to datetime object."""
    if num is None:
        return None
    # Handle both seconds and milliseconds
    unit = 's' if abs(num) < 1e11 else 'ms'
    return pd.to_datetime(num, unit=unit).to_pydatetime()

def date2num(dt):
    """Convert datetime object to numeric timestamp."""
    if dt is None:
        return None
    return pd.to_datetime(dt).timestamp()

# Aliases for Backtrader compatibility
az = analyzers
obs = observers
CommissionInfo = CommInfoBase
talib = importlib.import_module("tradelearn.indicators.talib")
tdx = importlib.import_module("tradelearn.indicators.tdx")
tv = importlib.import_module("tradelearn.indicators.tv")


def __getattr__(name: str) -> Any:
    if name == "MLStrategy":
        from tradelearn.ml import MLStrategy

        return MLStrategy
    raise AttributeError(f"module 'tradelearn.engine' has no attribute {name!r}")

__all__ = [
    "Cerebro",
    "OptReturn",
    "DataFeed",
    "CommInfoBase",
    "ExecutedInfo",
    "LineSeries",
    "Order",
    "Params",
    "Position",
    "Sizer",
    "Strategy",
    "MLStrategy",
    "Trade",
    "Indicator",
    "IndexEnhanceStrategy",
    "TimeFrame",
    "FixedSize",
    "PercentSizer",
    "AllInSizer",
    "Analyzer",
    "feeds",
    "analyzers",
    "az",
    "observers",
    "sizers",
    "GridSearchResult",
    "grid_search",
    "Observer",
    "obs",
    "CommissionInfo",
    "talib",
    "tdx",
    "tv",
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
