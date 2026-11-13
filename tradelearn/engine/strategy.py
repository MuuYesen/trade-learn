"""Backtrader-style strategy aliases for one-line import migration.

The core backtest API already follows backtrader's line convention:
``line[0]`` is the current bar and ``line[-1]`` is the previous bar. This
module exposes that implementation under ``tradelearn.engine``.
"""

from datetime import timedelta

from tradelearn.backtest.history import build_history_panel
from tradelearn.backtest.models import ExecutedInfo, Position, Trade
from tradelearn.engine.order import Order
from tradelearn.backtest.strategy import Strategy as _BaseStrategy
from tradelearn.core.strategy_events import (
    assert_observer_allows_mutation,
    capture_observation_clock,
)
from tradelearn.engine.base import _G, LineRoot, LineSeries, Params, MetaParams, collect_param_defaults, set_current_data
from tradelearn.engine.datafeed import DataFeed
from tradelearn.engine.sizers import Sizer


class Strategy(_BaseStrategy, LineRoot):
    """Backtrader-compatible Strategy that maintains data context."""

    params = ()

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
    _bt_minperiods = True

    def _batch_runtime(self):
        runtime = getattr(self, "_runtime", None)
        if runtime is None:
            runtime = getattr(getattr(self, "data", None), "runtime", None)
        return runtime if runtime is not None and runtime.profile == "batch" else None

    def __len__(self):
        runtime = getattr(self, "_runtime", None)
        if runtime is None:
            runtime = getattr(getattr(self, "data", None), "runtime", None)
        if runtime is None:
            return super().__len__()
        return getattr(runtime, '_strategy_cursor', runtime._cursor) + 1 if runtime.profile == "batch" else runtime._sequence

    @property
    def datetime(self):
        runtime = self._batch_runtime()
        if runtime is None:
            owner = getattr(self, "_runtime", None)
            if owner is None:
                owner = getattr(getattr(self, "data", None), "runtime", None)
            clock = getattr(owner, "_strategy_datetime_line", None)
            if clock is not None:
                return clock
            return self.data.datetime
        line = getattr(self, "_clock_datetime_line", None)
        if line is None:
            from tradelearn.engine.datafeed import _DateLine
            clock_feed = runtime._clock_feed
            replay = getattr(runtime._clock,'strategy_lengths',None) is not None
            values = clock_feed._buffer.arrays['datetime'] if replay else runtime._clock.unixseconds
            line = _DateLine(values, is_datetime=True)
            line._cursor_source, line._runtime = (clock_feed if replay else runtime), runtime
            if runtime._clock.unit == 'ns':
                line._epoch_ns_values = clock_feed._buffer._epoch_ns_values if replay else runtime._clock.timestamps
            self._clock_datetime_line = line
        return line

    def add_timer(self, when, offset=timedelta(), repeat=timedelta(),
                  weekdays=[], weekcarry=False, monthdays=[], monthcarry=True,
                  allow=None, tzdata=None, cheat=False, *args, **kwargs):
        """Register a BT-owned timer; startup prepares its bar-clock state."""
        assert_observer_allows_mutation()
        return self.cerebro._add_timer(owner=self,when=when,offset=offset,repeat=repeat,
            weekdays=weekdays,weekcarry=weekcarry,monthdays=monthdays,monthcarry=monthcarry,
            allow=allow,tzdata=tzdata,strats=False,cheat=cheat,*args,**kwargs)

    def notify_timer(self,timer,when,*args,**kwargs):
        pass

    def addminperiod(self, minperiod):
        """Apply BT line minimums without overriding per-data eligibility.

        BT Strategy._periodset derives callback readiness from its feeds and
        indicators. Direct Strategy.addminperiod affects its own lines only;
        Lite retains its separate convenient strategy-wide warmup control.
        """
        assert_observer_allows_mutation()
        for line in self.lines:
            if line is not None:
                line.min_period = getattr(line, 'min_period', 1) + minperiod - 1

    def getposition(self, data=None, broker=None):
        return (broker or self.broker).getposition(self._resolve_data(data))

    def getpositionbyname(self, name=None, broker=None):
        data = self.getdatabyname(name) if name else self.datas[0]
        return self.getposition(data,broker)

    positionbyname = property(getpositionbyname)

    def getpositions(self, broker=None):
        return (broker or self.broker).positions

    positions = property(getpositions)

    def getpositionsbyname(self, broker=None):
        broker = broker or self.broker
        return {data._name:broker.getposition(data) for data in self.datas}

    positionsbyname = property(getpositionsbyname)

    def order_target_size(self, data=None, target=0, **kwargs):
        data = self._resolve_data(data)
        position = self.getposition(data, self.broker).size
        if not target and position:
            return self.close(data=data, size=position, **kwargs)
        if target > position:
            return self.buy(data=data, size=target-position, **kwargs)
        if target < position:
            return self.sell(data=data, size=position-target, **kwargs)
        return None

    def order_target_value(self, data=None, target=0.0, price=None, **kwargs):
        data = self._resolve_data(data)
        position = self.getposition(data, self.broker).size
        if not target and position:
            return self.close(data=data, size=position, price=price, **kwargs)
        value = self.broker.getvalue(datas=[data])
        comminfo = self.broker.getcommissioninfo(data)
        price = data.close[0] if price is None else price
        if target > value:
            return self.buy(data=data, size=comminfo.getsize(price,target-value),price=price,**kwargs)
        if target < value:
            return self.sell(data=data, size=comminfo.getsize(price,value-target),price=price,**kwargs)
        return None

    def setsizer(self, sizer):
        self._sizer = sizer
        sizer.set(self, self.broker)
        return sizer

    def close(self, data=None, size=None, **kwargs):
        data = self._resolve_data(data)
        if size is not None:
            size = abs(size)
            if not size:
                return None
        close = getattr(self.broker, '_close_position', None)
        if callable(close):
            result = close(self, data, size=size, **kwargs)
            if result is not NotImplemented:
                return result
        return super().close(data=data, size=size, **kwargs)

    def buy(self, data=None, size=None, price=None, plimit=None, exectype=None,
            valid=None, tradeid=0, oco=None, trailamount=None, trailpercent=None,
            parent=None, transmit=True, **kwargs):
        return self._submit_bt(Order.Buy, data, size, price, plimit, exectype,
                               valid, tradeid, oco, trailamount, trailpercent, parent, transmit, kwargs)

    def sell(self, data=None, size=None, price=None, plimit=None, exectype=None,
             valid=None, tradeid=0, oco=None, trailamount=None, trailpercent=None,
             parent=None, transmit=True, **kwargs):
        return self._submit_bt(Order.Sell, data, size, price, plimit, exectype,
                               valid, tradeid, oco, trailamount, trailpercent, parent, transmit, kwargs)

    def _submit_bt(self, side, data, size, price, plimit, exectype, valid, tradeid,
                   oco, trailamount, trailpercent, parent, transmit, controls):
        assert_observer_allows_mutation()
        entry_clock = capture_observation_clock(self, "intent.submit_order")
        data = self._resolve_data(data)
        size = abs(self.getsizing(data, isbuy=side == Order.Buy) if size is None else size)
        if not size:
            return None
        if valid is not None:
            import datetime
            import math
            import numbers
            from tradelearn.backtest.clock import date2num
            if isinstance(valid, datetime.timedelta):
                now = data.datetime.datetime(0)
                if now is None:
                    raise ValueError("order validity requires a delivered data clock")
                # Fixed BT DAY is a timedelta sentinel, distinct from numeric0.
                deadline = (now.replace(hour=23, minute=59, second=59, microsecond=9999)
                            if valid == datetime.timedelta(0) else now + valid)
                valid = date2num(deadline)
            elif isinstance(valid, datetime.date):
                valid = date2num(valid)
            elif isinstance(valid, numbers.Real):
                if not math.isfinite(float(valid)):
                    raise ValueError("finite BT ordinal validity required")
                # Fixed BT retains numeric values literally:0 is unbounded,
                # other values are absolute ordinal days, never relative seconds.
            else:
                raise TypeError("valid must be BT ordinal days, date, datetime or timedelta")
        if tradeid != 0:
            raise NotImplementedError("multiple tradeid accounting is not implemented")
        if plimit is not None:
            if "pricelimit" in controls:
                raise ValueError("plimit and pricelimit conflict")
            controls["pricelimit"] = plimit
        for name, value in dict(valid=valid, oco=oco, trailamount=trailamount,
                                trailpercent=trailpercent, parent=parent).items():
            if value is not None:
                controls[name] = value
        if not transmit:
            controls["transmit"] = transmit
        return super()._submit_order_observed(entry_clock, side, self._resolve_data(data), size, price, exectype, **controls)

    def _resolve_data(self, data=None):
        if isinstance(data, str):
            return self.getdatabyname(data)
        return super()._resolve_data(data)

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





__all__ = [
    "DataFeed",
    "ExecutedInfo",
    "LineSeries",
    "Order",
    "Params",
    "Position",
    "Sizer",
    "Strategy",
    "Trade",
]
