"""Backtrader-compatible facade backed by tradelearn's backtest engine."""

from tradelearn.compat.backtrader import feeds, indicators, analyzers
from tradelearn.compat.backtrader.indicators import Indicator
from tradelearn.compat.backtrader.cerebro import Cerebro
from tradelearn.compat.backtrader.strategy import (
    DataFeed,
    CommInfoBase,
    ExecutedInfo,
    LineSeries,
    Order,
    Params,
    Position,
    Sizer,
    Strategy,
    Trade,
)
import pandas as pd

def num2date(num):
    """Convert numeric timestamp back to datetime object."""
    if num is None: return None
    # Handle both seconds and milliseconds
    unit = 's' if abs(num) < 1e11 else 'ms'
    return pd.to_datetime(num, unit=unit).to_pydatetime()

def date2num(dt):
    """Convert datetime object to numeric timestamp."""
    if dt is None: return None
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
    "feeds",
    "indicators",
    "ind",
    "analyzers",
    "az",
    "num2date",
    "date2num",
]
