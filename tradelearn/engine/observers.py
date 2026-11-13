from __future__ import annotations

from typing import Any

from tradelearn.engine.base import (BaseObserver, MetaParams, Params,
                                    collect_param_defaults, _bind_component_data_aliases)


class ObserverCollection:
    """BT ordered items permit duplicate Observer names.

    Index/getbyname address the first registration; attribute lookup exposes
    the most recently appended same-name observer, as fixed BT does.
    AnalyzerCollection remains a separate unique-name/suffix protocol.
    """
    def __init__(self, source=()):
        self._names, self._items, self._latest = [], [], {}
        entries = source.items() if hasattr(source, 'items') else source
        for name, value in entries:
            self[name] = value

    def __setitem__(self, name, value):
        self._names.append(name)
        self._items.append(value)
        self._latest[name] = value

    def __getitem__(self, key):
        return self._items[key] if isinstance(key, (int, slice)) else self.getbyname(key)

    def __getattr__(self, name):
        latest = vars(self).get('_latest', {})
        try:
            return latest[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __contains__(self, name):
        return name in self._names

    def getnames(self):
        return list(self._names)

    def getitems(self):
        return zip(self._names, self._items)

    items = getitems
    keys = getnames

    def values(self):
        return list(self._items)

    def getbyname(self, name):
        return self._items[self._names.index(name)]

    def get(self, name, default=None):
        return self.getbyname(name) if name in self else default

    def setdefault(self, name, value):
        if name not in self:
            self[name] = value
        return self.getbyname(name)


class Observer(BaseObserver, metaclass=MetaParams):
    """Minimal Backtrader observer base class."""

    params: tuple = ()
    _stclock = False

    def _base_init(self, **kwargs: Any) -> None:
        self.params = self.p = Params(collect_param_defaults(type(self))[1], **kwargs)
        self.strategy = None
        self.values: list[Any] = []
        from tradelearn.backtest.lines import Lines
        self.lines = self.l = Lines(self)
        self._line_clock_feed = None
        self._line_last_count = 0
        self._minperiod = 1

    def __init__(self) -> None:
        pass

    def _initialize_component_context(self, strategy: Any) -> None:
        self._owner = self._clock = self.strategy = strategy
        _bind_component_data_aliases(self)
        from tradelearn.backtest.lines import line_names
        self._runonce = getattr(strategy, "_observer_runonce", True)
        runtime = getattr(strategy, "_runtime", None)
        if runtime is not None:
            clock = (getattr(runtime, "_clock_feed", None)
                     if self._runonce or self._stclock else None)
            self._line_clock_feed = clock if clock is not None else runtime._feed_for(self.data)
            self._line_last_count = self._line_clock_feed.received
            for name in line_names(type(self)):
                setattr(self.lines, name, runtime.output_line(self._line_clock_feed))

    def __getattr__(self, name):
        lines = vars(self).get("lines")
        if lines is not None and not name.startswith("_"):
            line = vars(lines).get(name)
            if line is not None:
                return line
        raise AttributeError(name)

    def __getitem__(self, ago):
        return self.lines[0][ago]

    def addminperiod(self, minperiod):
        # BT adds each request minus its overlapping first sample.
        for line in self.lines:
            line.min_period = max(1, line.min_period) + minperiod - 1
        self._minperiod = max((line.min_period for line in self.lines), default=1)

    def _next_step(self):
        # Observe on the existing market/clock; never register a new indicator
        # or make observer outputs change strategy eligibility.
        feed = self._line_clock_feed
        if feed is not None:
            for _ in range(feed.received - self._line_last_count):
                for line in self.lines:
                    line._append_output()
            self._line_last_count = feed.received
        runtime = getattr(self.strategy, "_runtime", None)
        phase = (runtime._next_callback(not runtime._warming_up())
                 if runtime is not None and getattr(runtime, "_bt_line_views", False) else "next")
        if runtime is not None and not self._runonce:
            # Event Observer owns its minperiod; shared graph still performs all
            # calculations. Business eligibility and callbacks stay independent.
            minimum = self._minperiod
            for line in (*runtime._indicators, *runtime._plans.values(),
                         *getattr(runtime, "_compiled_lines", {}).values()):
                if getattr(line, "_eligibility_owner", None) is self:
                    minimum = max(minimum, int(getattr(line, "_minperiod",
                                                       getattr(line, "min_period", 1))))
            count = feed.received if feed is not None else len(self.strategy)
            phase = "prenext" if count < minimum else "nextstart" if count == minimum else "next"
        getattr(self, phase)()

    def __len__(self) -> int:
        if not getattr(self, "_runonce", True) and self._line_clock_feed is not None:
            return self._line_clock_feed.received
        return len(self.strategy)

    def _set(self, strategy: Any) -> None:
        self.strategy = strategy

    def start(self) -> None:
        pass

    def prenext(self) -> None:
        self.next()

    def nextstart(self) -> None:
        self.next()

    def next(self) -> None:
        self.values.append(self._observe())

    def _stop(self) -> None:
        pass

    def _observe(self) -> Any:
        return None

    def _get_analysis(self) -> dict[str, Any]:
        return {self.__class__.__name__.lower(): self.values}


class _AccountValueProfile:
    params = (('fund', None),)
    _stclock = True

    def start(self):
        broker = self.strategy.broker
        self._fundmode = getattr(broker, 'fundmode', False) if self.p.fund is None else self.p.fund
        if self._fundmode and not hasattr(broker, 'fundvalue'):
            raise NotImplementedError('fund Observer requires authoritative broker fundvalue')

    def _value(self):
        broker = self.strategy.broker
        return float(broker.fundvalue if self._fundmode else broker.getvalue())

class Value(_AccountValueProfile, Observer):
    lines = ('value',)

    def _observe(self):
        value = self._value()
        self.lines.value[0] = value
        return value

    def _get_analysis(self):
        return {"value": self.values}


class Broker(_AccountValueProfile, Observer):
    lines = ('cash', 'value')
    _get_analysis = Value._get_analysis

    def _observe(self):
        value = self._value()
        self.lines.value[0] = value
        if not self._fundmode:
            self.lines.cash[0] = float(self.strategy.broker.getcash())
        return value


class BuySell(Observer):
    lines = ('buy', 'sell')
    params = (('barplot', False), ('bardist', .015))
    _observes_orders = True

    def __init__(self):
        self._pending_buy = [0., 0]
        self._pending_sell = [0., 0]
        self._buy_count = self._sell_count = 0

    def _notify_order(self, order):
        if order.data is not self.data or not order.executed.size:
            return
        bucket = self._pending_buy if order.isbuy() else self._pending_sell
        bucket[0] += order.executed.price
        bucket[1] += 1

    def _observe(self):
        import math
        for name, bucket in (('buy', self._pending_buy), ('sell', self._pending_sell)):
            line = getattr(self.lines, name)
            previous = float(line[0])
            old_count = getattr(self, '_' + name + '_count') if not math.isnan(previous) else 0
            count = old_count + bucket[1]
            numerator = (0. if math.isnan(previous) else previous) + bucket[0]
            value = numerator / count if count else math.nan
            if self.p.barplot and not math.isnan(value):
                value = (self.data.low[0] * (1 - self.p.bardist) if name == 'buy'
                         else self.data.high[0] * (1 + self.p.bardist))
            line[0] = value
            setattr(self, '_' + name + '_count', count)
            bucket[:] = [0., 0]
        compat = getattr(self, '_compat_count', None)
        return compat() if compat is not None else len(getattr(self.strategy.broker, '_fills', ()))

    def _get_analysis(self):
        return {"fills": self.values}


class Trades(Observer):
    lines = ('pnlplus', 'pnlminus')
    params = (('pnlcomm', True),)
    _stclock = True
    _observes_trades = True

    def __init__(self):
        self._pending_plus = self._pending_minus = None

    def _notify_trade(self, trade):
        if not trade.isclosed or not any(trade.data is data for data in self.datas):
            return
        value = trade.pnlcomm if self.p.pnlcomm else trade.pnl
        if value >= 0:
            self._pending_plus = value
        else:
            self._pending_minus = value

    def _observe(self):
        if self._pending_plus is not None:
            self.lines.pnlplus[0] = self._pending_plus
        if self._pending_minus is not None:
            self.lines.pnlminus[0] = self._pending_minus
        self._pending_plus = self._pending_minus = None
        compat = getattr(self, '_compat_count', None)
        return compat() if compat is not None else len(getattr(self.strategy, '_trades', ()))

    def _get_analysis(self):
        return {"trades": self.values}


class DrawDown(_AccountValueProfile, Observer):
    lines = ('drawdown', 'maxdrawdown')

    def start(self):
        super().start()
        self._peak = None
        self._maximum = 0.
        self._pending_drawdown = None

    def _before_business(self):
        from tradelearn.metrics.risk import _drawdown_observation
        self._peak, ratio, _ = _drawdown_observation(self._value(), self._peak)
        self._pending_drawdown = -100. * ratio

    def _observe(self):
        value = self._pending_drawdown
        if value is None:
            self._before_business()
            value = self._pending_drawdown
        self._maximum = max(self._maximum, value)
        self.lines.drawdown[0], self.lines.maxdrawdown[0] = value, self._maximum
        self._pending_drawdown = None
        return value

    def _get_analysis(self):
        return {"drawdown": self.values}


class TimeReturn(_AccountValueProfile, Observer):
    lines = ('timereturn',)
    params = (('timeframe', None), ('compression', None), ('fund', None))

    def start(self):
        super().start()
        from tradelearn.engine.analyzers.returns import _validate_period
        self._timeframe = self.p.timeframe or self.data._timeframe
        self._compression = self.p.compression or self.data._compression
        _validate_period(self._timeframe, self._compression)
        self._previous_value = self._value()
        self._period = None
        self._period_start = self._previous_value
        self._pending_return = None

    def _before_business(self):
        from tradelearn.engine.analyzers.returns import _period_key
        key = _period_key(self.strategy.datetime.datetime(), self._timeframe, self._compression)
        if self._period is None or key > self._period:
            self._period_start = self._previous_value
            self._period = key
        value = self._value()
        self._pending_return = value / self._period_start - 1.
        self._previous_value = value

    def _observe(self):
        if self._pending_return is None:
            self._before_business()
        value = self._pending_return
        self.lines.timereturn[0] = value
        self._pending_return = None
        return value

    def _get_analysis(self):
        return {"timereturn": self.values}


__all__ = [
    "Broker",
    "BuySell",
    "DrawDown",
    "Observer",
    "ObserverCollection",
    "TimeReturn",
    "Trades",
    "Value",
]
