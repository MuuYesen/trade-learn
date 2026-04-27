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
    Strategy as _BaseStrategy,
    Trade,
)
from tradelearn.compat.backtrader.indicators import set_current_data

class Strategy(_BaseStrategy):
    """Backtrader-compatible Strategy that maintains data context."""
    def __init__(self, *args, **kwargs):
        # Set context for indicators created during __init__
        if self.datas:
            set_current_data(self.datas[0])
        super().__init__(*args, **kwargs)
        # Clear context after init
        set_current_data(None)
        
    @property
    def datetime(self):
        """Shortcut for self.data.datetime to match backtrader behavior."""
        return self.data.datetime



class Sizer:
    """Base class for strategy sizing logic."""
    def __init__(self):
        pass

    def _getsizing(self, comminfo, cash, data, isbuy):
        raise NotImplementedError


class CommInfoBase:
    """Base class for commission schemes."""
    COMM_PERC = 0
    COMM_CASH = 1
    params = ()
    def __init__(self, *args, **kwargs):
        pass




__all__ = [
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
]
