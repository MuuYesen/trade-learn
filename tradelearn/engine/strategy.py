"""Backtrader-style strategy aliases for one-line import migration.

The core backtest API already follows backtrader's line convention:
``line[0]`` is the current bar and ``line[-1]`` is the previous bar. This
module exposes that implementation under ``tradelearn.engine``.
"""

from tradelearn.backtest.history import build_history_panel
from tradelearn.backtest.models import ExecutedInfo, Order, Position, Trade
from tradelearn.backtest.strategy import Strategy as _BaseStrategy
from tradelearn.engine.base import _G, LineRoot, LineSeries, Params, set_current_data
from tradelearn.engine.datafeed import DataFeed
from tradelearn.engine.sizers import Sizer


class Strategy(_BaseStrategy, LineRoot):
    """Backtrader-compatible Strategy that maintains data context."""

    params = (("research_result", None),)

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
        research_result = getattr(getattr(self, "p", None), "research_result", None)
        if research_result is not None:
            self.research_result = research_result

    def _bind_research_result(self, result):
        from tradelearn.research.run import bind_research_result

        return bind_research_result(result, self)
        
    @property
    def datetime(self):
        """Shortcut for self.data.datetime to match backtrader behavior."""
        return self.data.datetime

    def history_panel(self, lookback: int | None = None):
        """Return recent multi-data OHLCV bars as a timestamp/symbol panel."""

        return build_history_panel(self.datas, lookback=lookback)

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
        return super().buy_bracket(
            data=data,
            size=size,
            price=price,
            stopprice=stopprice,
            limitprice=limitprice,
            pricelimit=plimit,
            exectype=exectype,
            valid=valid,
            trailamount=trailamount,
            trailpercent=trailpercent,
            stopexec=stopexec,
            limitexec=limitexec,
            oargs=oargs,
            stopargs=stopargs,
            limitargs=limitargs,
            **kwargs,
        )

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
        return super().sell_bracket(
            data=data,
            size=size,
            price=price,
            stopprice=stopprice,
            limitprice=limitprice,
            pricelimit=plimit,
            exectype=exectype,
            valid=valid,
            trailamount=trailamount,
            trailpercent=trailpercent,
            stopexec=stopexec,
            limitexec=limitexec,
            oargs=oargs,
            stopargs=stopargs,
            limitargs=limitargs,
            **kwargs,
        )



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
