"""Backtrader-style strategy aliases for one-line import migration.

The core backtest API already follows backtrader's line convention:
``line[0]`` is the current bar and ``line[-1]`` is the previous bar. This
module exposes that implementation under ``tradelearn.compat.backtrader``.
"""

from tradelearn.backtest import (
    DataFeed,
    ExecutedInfo,
    LineSeries,
    Order,
    Params,
    Position,
    Strategy,
    Trade,
)

__all__ = [
    "DataFeed",
    "ExecutedInfo",
    "LineSeries",
    "Order",
    "Params",
    "Position",
    "Strategy",
    "Trade",
]
