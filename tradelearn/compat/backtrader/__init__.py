"""Backtrader-compatible facade backed by tradelearn's backtest engine."""

from tradelearn.compat.backtrader import feeds, indicators
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

# Aliases for Backtrader compatibility
ind = indicators

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
]
