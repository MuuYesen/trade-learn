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
from tradelearn.compat.backtrader.indicators import set_current_data, set_current_strategy

class Strategy(_BaseStrategy):
    """Backtrader-compatible Strategy that maintains data context."""
    def __init__(self, *args, **kwargs):
        # Set context for indicators created during __init__
        if self.datas:
            set_current_data(self.datas[0])
        set_current_strategy(self)
        super().__init__(*args, **kwargs)
        # Clear context after init
        set_current_data(None)
        set_current_strategy(None)
        
    @property
    def datetime(self):
        """Shortcut for self.data.datetime to match backtrader behavior."""
        return self.data.datetime

    def order_target_percent(self, target: float = 0.0, data: None = None):
        """Standard Backtrader order_target_percent implementation."""
        from typing import Any
        if data is None:
            data = self.data
            
        value = self.broker.getvalue()
        target_value = value * target
        price = data.close[0]
        
        current_pos = self.getposition(data).size
        target_size = int(target_value / price)
        diff = target_size - current_pos
        
        if diff > 0:
            return self.buy(data=data, size=diff)
        elif diff < 0:
            return self.sell(data=data, size=abs(diff))
        return None



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
