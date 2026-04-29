"""Backtrader-compatible facade backed by tradelearn's backtest engine."""

import pandas as pd

from tradelearn.engine import analyzers, feeds, indicators, observers, sizers
from tradelearn.engine.analyzer import Analyzer
from tradelearn.engine.base import TimeFrame
from tradelearn.engine.cerebro import Cerebro
from tradelearn.engine.grid import GridSearchResult, grid_search
from tradelearn.engine.indicators import Indicator
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
ind = indicators
az = analyzers

__all__ = [
    "Cerebro",
    "DataFeed",
    "CommInfoBase",
    "ExecutedInfo",
    "LineSeries",
    "Order",
    "Params",
    "Position",
    "Sizer",
    "Strategy",
    "Trade",
    "Indicator",
    "TimeFrame",
    "FixedSize",
    "PercentSizer",
    "AllInSizer",
    "Analyzer",
    "feeds",
    "indicators",
    "ind",
    "analyzers",
    "az",
    "observers",
    "sizers",
    "GridSearchResult",
    "grid_search",
    "num2date",
    "date2num",
]
