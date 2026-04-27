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
from tradelearn.backtest.base import set_current_data, set_current_strategy

class Strategy(_BaseStrategy):
    """Backtrader-compatible Strategy that maintains data context."""
    def __init__(self, *args, **kwargs):
        # Set context for indicators created during __init__
        if self.datas:
            set_current_data(self.datas[0])
        super().__init__(*args, **kwargs)
        
    @property
    def datetime(self):
        """Shortcut for self.data.datetime to match backtrader behavior."""
        return self.data.datetime

    def order_target_size(self, data=None, target=0, **kwargs):
        """Place an order to rebalance position to have final size of target."""
        if data is None:
            data = self.data
        # Use the effective position (including pending orders) to avoid duplicate orders
        possize = self.getposition(data).size + self._pending_size.get(data, 0.0)
        if not target and possize:
            return self.close(data=data, **kwargs)
        elif target > possize:
            return self.buy(data=data, size=target - possize, **kwargs)
        elif target < possize:
            return self.sell(data=data, size=possize - target, **kwargs)
        return None

    def order_target_value(self, data=None, target=0.0, price=None, **kwargs):
        """Place an order to rebalance position to have final value of target."""
        if data is None:
            data = self.data
        # Use effective position to account for pending rebalances
        possize = self.getposition(data).size + self._pending_size.get(data, 0.0)
        if not target and possize:
            return self.close(data=data, **kwargs)
        price = price if price is not None else data.close[0]
        # Get multiplier from commission info if possible
        comminfo = self.broker.getcommissioninfo(data)
        mult = getattr(comminfo.p, 'mult', 1.0)
        
        current_value = possize * price * mult
        if target > current_value:
            size = int((target - current_value) / (price * mult))
            if size:
                return self.buy(data=data, size=size, price=price, **kwargs)
        elif target < current_value:
            size = int((current_value - target) / (price * mult))
            if size:
                return self.sell(data=data, size=size, price=price, **kwargs)
        return None

    def order_target_percent(self, data=None, target=0.0, **kwargs):
        """Place an order to rebalance position to target percentage of portfolio value."""
        if data is None:
            data = self.data
        target_value = target * self.broker.getvalue()
        return self.order_target_value(data=data, target=target_value, **kwargs)



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
