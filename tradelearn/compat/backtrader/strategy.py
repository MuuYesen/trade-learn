"""Backtrader-style strategy aliases for one-line import migration.

The core backtest API already follows backtrader's line convention:
``line[0]`` is the current bar and ``line[-1]`` is the previous bar. This
module exposes that implementation under ``tradelearn.compat.backtrader``.
"""

from tradelearn.backtest.models import ExecutedInfo, Order, Position, Trade
from tradelearn.backtest.strategy import Strategy as _BaseStrategy
from tradelearn.compat.backtrader.base import _G, LineRoot, LineSeries, Params, set_current_data
from tradelearn.compat.backtrader.datafeed import DataFeed
from tradelearn.compat.backtrader.sizers import Sizer


class Strategy(_BaseStrategy, LineRoot):
    """Backtrader-compatible Strategy that maintains data context."""
    def __init__(self, *args, **kwargs):
        # 1. Sync datas from global context if not already set (for BT facade)
        if not getattr(self, 'datas', None) and _G.current_datas:
            self.datas = _G.current_datas
            self.data = self.datas[0] if self.datas else None
            for i, d in enumerate(self.datas):
                setattr(self, f'data{i}', d)

        # 2. Set context for indicators created during __init__
        if self.datas:
            set_current_data(self.datas[0])
        
        # 3. Call base inits
        # Note: _BaseStrategy.__init__ handles internal dicts
        # LineRoot handled by metaclass already, but we ensure order
        _BaseStrategy.__init__(self, *args, **kwargs)
        
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

    def cancel(self, order):
        return self.broker.cancel(order)

    def buy_bracket(
        self,
        data=None,
        size=None,
        price=None,
        plimit=None,
        exectype=Order.Limit,
        valid=None,
        trailamount=None,
        trailpercent=None,
        oargs=None,
        stopprice=None,
        stopexec=Order.Stop,
        stopargs=None,
        limitprice=None,
        limitexec=Order.Limit,
        limitargs=None,
        **kwargs,
    ):
        data = data or self.data
        oargs = dict(oargs or {})
        stopargs = dict(stopargs or {})
        limitargs = dict(limitargs or {})
        main = self.buy(
            data=data,
            size=size,
            price=price,
            pricelimit=plimit,
            exectype=exectype,
            valid=valid,
            transmit=False,
            **kwargs,
            **oargs,
        )
        stop = self.sell(
            data=data,
            size=size,
            price=stopprice,
            exectype=stopexec,
            parent=main,
            transmit=False,
            trailamount=trailamount,
            trailpercent=trailpercent,
            **stopargs,
        )
        limit = self.sell(
            data=data,
            size=size,
            price=limitprice,
            exectype=limitexec,
            parent=main,
            oco=stop,
            transmit=True,
            **limitargs,
        )
        return [main, stop, limit]

    def sell_bracket(
        self,
        data=None,
        size=None,
        price=None,
        plimit=None,
        exectype=Order.Limit,
        valid=None,
        trailamount=None,
        trailpercent=None,
        oargs=None,
        stopprice=None,
        stopexec=Order.Stop,
        stopargs=None,
        limitprice=None,
        limitexec=Order.Limit,
        limitargs=None,
        **kwargs,
    ):
        data = data or self.data
        oargs = dict(oargs or {})
        stopargs = dict(stopargs or {})
        limitargs = dict(limitargs or {})
        main = self.sell(
            data=data,
            size=size,
            price=price,
            pricelimit=plimit,
            exectype=exectype,
            valid=valid,
            transmit=False,
            **kwargs,
            **oargs,
        )
        stop = self.buy(
            data=data,
            size=size,
            price=stopprice,
            exectype=stopexec,
            parent=main,
            transmit=False,
            trailamount=trailamount,
            trailpercent=trailpercent,
            **stopargs,
        )
        limit = self.buy(
            data=data,
            size=size,
            price=limitprice,
            exectype=limitexec,
            parent=main,
            oco=stop,
            transmit=True,
            **limitargs,
        )
        return [main, stop, limit]



class CommInfoBase:
    """Base class for commission schemes."""
    COMM_PERC = 0
    COMM_CASH = 1
    params = (
        ("commission", 0.0),
        ("mult", 1.0),
        ("margin", None),
        ("commtype", COMM_PERC),
        ("stocklike", True),
        ("percabs", True),
    )

    def __init__(self, *args, **kwargs):
        self.params = self.p = Params(self.params, **kwargs)

    def getcommission(self, size, price, *args):
        if self.p.commtype == self.COMM_CASH:
            return abs(size) * self.p.commission
        return abs(size) * price * self.p.commission * self.p.mult

    def get_margin(self, price):
        return self.p.margin if self.p.margin is not None else price * self.p.mult

    def getsize(self, price, cash):
        margin = self.get_margin(price)
        if margin <= 0:
            return 0
        return int(cash / margin)

    def getoperationcost(self, size, price):
        return abs(size) * price * self.p.mult

    def getvaluesize(self, size, price):
        return size * price * self.p.mult

    def profitandloss(self, size, price, newprice):
        return size * (newprice - price) * self.p.mult


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
