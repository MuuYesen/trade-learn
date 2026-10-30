"""Synchronous, broker-neutral incremental strategy lifecycle.

Callers own scheduling and authoritative broker state. Each market step exposes
only the received per-symbol prefix; no matching engine or background loop runs.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
import math
import warnings
from dataclasses import dataclass
from threading import get_ident
from types import MappingProxyType, SimpleNamespace
from weakref import WeakValueDictionary

import numpy as np
import pandas as pd

from tradelearn.backtest.data import RollingBarBuffer
from tradelearn.backtest.lines import Lines, LineSeries
from tradelearn.backtest.strategy import Strategy
from tradelearn.core.strategy_context import engine_context


def _safe_cleanup_diagnostic(primary_error, cleanup_error):
    """Best-effort diagnostics must never replace the original exception.

    Python 3.10 has no BaseException.add_note. Custom error renderers, note
    methods and warning handlers are also user code and can fail themselves.
    """
    try:
        try:
            detail = str(cleanup_error)
        except BaseException:
            detail = "<unprintable cleanup exception>"
        message = (
            "TradeLearn cleanup failed while preserving the primary exception: "
            + type(cleanup_error).__name__ + ": " + detail
        )
    except BaseException:
        message = "TradeLearn cleanup failed: <unprintable cleanup exception>"
    try:
        add_note = getattr(primary_error, "add_note", None)
        if callable(add_note):
            add_note(message)
            return
    except BaseException:
        pass
    try:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    except BaseException:
        pass  # Warning handlers/filters cannot override the primary failure.


def _function_minimum(function, args, kwargs):
    """Shared inferred lookback for vector compilation and prefix plans."""
    bound = inspect.signature(function._func).bind_partial(*args, **kwargs)
    bound.apply_defaults()
    lookback = max([1] + [int(bound.arguments[k])
        for k in ("period", "length", "window", "timeperiod")
        if isinstance(bound.arguments.get(k), int)])
    name = getattr(function, "name", "")
    if name == "talib.MACD":
        fast = int(bound.arguments["fastperiod"])
        slow = int(bound.arguments["slowperiod"])
        signal = int(bound.arguments["signalperiod"])
        lookback = max(fast, slow) + signal - 1
    elif name in {"talib.RSI", "talib.ATR"}:
        lookback = int(bound.arguments["timeperiod"]) + 1
    dependency = max([0] + [int(getattr(value, "min_period", 0))
                           for value in (*args, *kwargs.values())])
    return max(1, dependency) + lookback - 1


@dataclass(frozen=True)
class RuntimeContext:
    symbol: str
    event: object
    ts_event_ns: int
    sequence: int


class RuntimeLine(LineSeries):
    """A causal line. Public views are bounded; full-prefix graphs retain seeds."""

    def __init__(
        self, feed, *, field=None, compute=None, dependencies=(), min_period=0, history="window"
    ):
        self.feed = feed
        self._runtime = feed.runtime
        self._runtime._owner()
        self.field = field
        self.compute = compute
        self.dependencies = dependencies
        self.history = history
        self.min_period = min_period
        self._is_datetime = field == "datetime"
        self._cache_key = None
        self._cache = None
        self._revision = 0
        self._version_epoch = -1
        self._version_state = None
        self._version_id = 0
        self._full_ready = False
        count = len(feed) - (
            1 if self._runtime._stepping and feed._name in self._runtime._active_market_symbols else 0
        )
        self._output = [np.nan] * max(0, count)
        self._all_output = None
        if compute is not None:
            if len(self._runtime._plans) >= self._runtime.max_plans:
                raise RuntimeError("runtime graph plan limit reached")
            self._runtime._plans[id(self)] = self
        if history == "full":
            self.require_full()

    @property
    def _cursor(self):
        return len(self.feed) - 1

    @property
    def _values(self):
        return self.to_series().to_numpy(dtype=float)

    def _advance(self, cursor):
        pass

    def require_full(self):
        if self._full_ready:
            return
        self._runtime._owner()
        self.feed.retain_full()
        if self.history != "full":
            self._cache_key = None
            self._runtime._mutation_epoch += 1
        self.history = "full"
        for dependency in self.dependencies:
            if isinstance(dependency, RuntimeLine):
                dependency.require_full()
        if self.field is None and self.compute is None and self._all_output is None:
            self._all_output = [np.nan] * (self.feed.received - len(self._output)) + list(
                self._output
            )

        self._full_ready = True

    def full_series(self):
        self.require_full()
        if self.field is not None:
            return pd.Series(self.feed._history[self.field][: self.feed.received], dtype=float)
        if self.compute is None:
            return pd.Series(self._all_output, dtype=float)
        return self._evaluate().copy()

    def _evaluate(self):
        try:
            return self._compute()
        except BaseException:
            self._runtime._failed = True
            raise

    def _compute(self):
        key = self.version
        if key != self._cache_key:
            from tradelearn.core.strategy_events import assert_observer_allows_mutation
            assert_observer_allows_mutation()
            count = self.feed.received if self.history == "full" else len(self.feed)
            result = (
                self.compute()
                if count >= self.min_period
                else pd.Series(np.nan, index=range(count))
            )
            if isinstance(result, pd.DataFrame):
                raise ValueError(
                    "runtime prefix function requires one output Series; select columns explicitly"
                )
            values = pd.Series(result, dtype=float)
            if len(values) != count:
                raise ValueError("runtime prefix result must preserve input history length")
            self._cache = values.reset_index(drop=True)
            self._cache_key = key
        return self._cache

    @property
    def version(self):
        epoch = self._runtime._mutation_epoch
        if self._version_epoch != epoch:
            state = (
                self.feed.revision,
                self._revision,
                self.history,
                tuple(d.version for d in self.dependencies if isinstance(d, RuntimeLine)),
            )
            if state != self._version_state:
                self._version_state = state
                self._version_id += 1
            self._version_epoch = epoch
        return self._version_id

    def to_series(self):
        if self.field is not None:
            return pd.Series(
                self.feed._buffer.arrays[self.field][: len(self.feed)].copy(), dtype=float
            )
        if self.compute is None:
            return pd.Series(self._output, dtype=float)
        return (
            self._evaluate().iloc[-len(self.feed) :].reset_index(drop=True).copy()
            if len(self.feed)
            else pd.Series(dtype=float)
        )

    def __len__(self):
        return len(self.feed)

    def __getitem__(self, ago):
        if isinstance(ago, slice):
            return self.to_series().to_numpy()[ago]
        if not isinstance(ago, (int, np.integer)):
            return self
        if ago > 0:
            raise ValueError("runtime lines cannot address future observations")
        if self.field is not None:
            values = self.feed._buffer.arrays[self.field][:len(self.feed)]
        elif self.compute is None:
            values = self._output
        else:
            values = self._evaluate().to_numpy()[-len(self.feed):] if len(self.feed) else ()
        i = len(values) - 1 + int(ago)
        value = np.nan if i < 0 else values[i]
        return self._format_value(value) if self._is_datetime else value

    def __setitem__(self, ago, value):
        self._runtime._owner()
        if ago != 0 or self.compute is not None or self.field is not None:
            raise ValueError("only incremental indicator current output is writable")
        if not self._output:
            raise RuntimeError("indicator output has no current market step")
        self._output[-1] = float(value)
        self._revision += 1
        self._runtime._mutation_epoch += 1
        if self._all_output is not None:
            self._all_output[-1] = float(value)

    def _append_output(self):
        self._revision += 1
        self._runtime._mutation_epoch += 1
        self._output.append(np.nan)
        if len(self._output) > self.feed._buffer.capacity:
            self._output.pop(0)
        if self._all_output is not None:
            self._all_output.append(np.nan)

    def get(self, ago=0, size=1):
        if ago > 0:
            raise ValueError("runtime lines cannot address future observations")
        values = self.to_series().to_numpy()
        end = max(0, len(values) + int(ago))
        return values[max(0, end - int(size)) : end]

    def __call__(self, ago=0):
        if ago > 0:
            raise ValueError("runtime lines cannot address future observations")
        result = RuntimeLine(
            self.feed,
            compute=lambda: (
                self.full_series() if result.history == "full" else self.to_series()
            ).shift(-ago),
            dependencies=(self,),
            min_period=self.min_period + abs(ago),
        )
        return result

    def _math_op(self, other, op):
        target = other.lines[0] if hasattr(other, "lines") else other
        if self._runtime.profile == "batch" and isinstance(target, LineSeries):
            target = self._runtime.causal_line(target)
        if not isinstance(target, RuntimeLine) and not np.isscalar(target):
            raise ValueError("runtime arithmetic requires a scalar or tracked runtime line")
        if isinstance(target, RuntimeLine) and target.feed is not self.feed:
            raise ValueError("cross-feed line arithmetic needs explicit timestamp alignment")

        def compute():
            a = self.full_series() if result.history == "full" else self.to_series()
            b = (
                (target.full_series() if result.history == "full" else target.to_series())
                if isinstance(target, RuntimeLine)
                else target
            )
            return op(a, b)

        result = RuntimeLine(
            self.feed,
            compute=compute,
            dependencies=(self, target),
            min_period=max(self.min_period, getattr(target, "min_period", 0)),
        )
        return result

    def _runtime_function(self, function, args, kwargs, *, history="full"):
        if history not in ("full", "window"):
            raise ValueError("history must be full or window")

        def line_of(x):
            value = x.lines[0] if hasattr(x, "lines") else x
            if self._runtime.profile == "batch" and isinstance(value, LineSeries):
                return self._runtime.causal_line(value)
            return value

        inputs = tuple(line_of(x) for x in args)
        options = {k: line_of(v) for k, v in kwargs.items()}
        for value in (*inputs, *options.values()):
            if not isinstance(value, RuntimeLine) and value is not None and not np.isscalar(value):
                raise ValueError("prefix inputs require scalar parameters or tracked runtime lines")
            if isinstance(value, RuntimeLine):
                if value.feed is not self.feed:
                    raise ValueError("cross-feed prefix needs explicit timestamp alignment")

        def resolve(x):
            return (
                (x.full_series() if history == "full" else x.to_series())
                if isinstance(x, RuntimeLine)
                else x
            )

        def compute():
            return function.compute(
                *(resolve(a) for a in inputs), **{k: resolve(v) for k, v in options.items()}
            )

        minimum = _function_minimum(function, inputs, options)
        if minimum > self.feed._buffer.capacity:
            raise ValueError("indicator warmup exceeds retained runtime capacity")
        if len(self._runtime._plans) >= self._runtime.max_plans:
            raise RuntimeError("runtime graph plan limit reached")
        if history == "full":
            lines = [v for v in (*inputs, *options.values()) if isinstance(v, RuntimeLine)]
            for line in lines:
                if line.feed._history is None and line.feed.received > len(line.feed):
                    raise ValueError("full-prefix indicator required history was discarded")
            for line in lines:
                line.require_full()
        return RuntimeLine(
            self.feed,
            compute=compute,
            dependencies=(*inputs, *options.values()),
            min_period=minimum,
            history=history,
        )

    def wrap_indicator(self, values, name=None):
        raise ValueError(
            "runtime line requires a prefix function plan; use runtime.prefix or FunctionIndicator"
        )


class RuntimeFeed:
    def __init__(self, runtime, symbol, capacity, buffer=None):
        self.runtime = runtime
        self._name = symbol
        self._buffer = buffer if buffer is not None else RollingBarBuffer(capacity)
        self.revision = 0
        self.quote = None
        self._history_materializations = 0
        self.received = (
            len(buffer)
            if hasattr(buffer, "__len__")
            else (buffer.cursor + 1 if buffer is not None else 0)
        )
        self._history = None
        self.lines = self.l = Lines(self)
        for name in ("datetime", "open", "high", "low", "close", "volume"):
            line = RuntimeLine(self, field=name)
            setattr(self, name, line)
            setattr(self.lines, name, line)

    def retain_full(self):
        if self._history is None:
            if self.received > len(self):
                raise ValueError(
                    "full-prefix indicator cannot start after required history was discarded"
                )
            self._history = {
                n: self._buffer.arrays[n][: len(self)].tolist() for n in self._buffer.arrays
            }

    def __len__(self):
        return self._buffer.cursor + 1

    @property
    def _cursor(self):
        return self._buffer.cursor

    @property
    def _frame(self):
        self._history_materializations += 1
        n = len(self)
        return pd.DataFrame(
            {
                name: self._buffer.arrays[name][:n]
                for name in ("open", "high", "low", "close", "volume")
            },
            index=pd.to_datetime(self._buffer.arrays["datetime"][:n], unit="s", utc=True),
        )


class _BoundBuffer:
    """Cursor view of a backend's preloaded storage; owns no second window."""
    def __init__(self, data):
        self.data = data
        self.arrays = {}
        self._cursor = -1
        for name in ("datetime", "open", "high", "low", "close", "volume"):
            line = getattr(data, name, None)
            self.arrays[name] = line._values if line is not None else getattr(data, "_" + name)
        self.capacity = max(2, len(self.arrays["close"]))

    @property
    def cursor(self):
        return getattr(self.data, "_cursor", self._cursor)


class BoundFeed(RuntimeFeed):
    def __init__(self, runtime, symbol, data):
        self.runtime, self._name, self.source = runtime, symbol, data
        self._buffer = _BoundBuffer(data)
        self.revision = 0
        self.quote = None
        self._history_materializations = 0
        self._history = None
        self.lines = self.l = Lines(self)
        for name in self._buffer.arrays:
            line = RuntimeLine(self, field=name)
            setattr(self, name, line)
            setattr(self.lines, name, line)

    @property
    def received(self):
        return self._buffer.cursor + 1

    def retain_full(self):
        if self._history is None:
            self._history = self._buffer.arrays


class StrategyRuntime:
    """One owner thread, one strategy, explicit ordered market/broker steps.

    ``bar`` events require timezone-aware ts and valid OHLCV. ``tick`` events
    retain the original object and use ts_event in nanoseconds, without bars.
    Equal timestamps are separate observations in call order. Late bars fail;
    tick event timestamps may be late, preserving source safety diagnostics.

    The default ``stream`` profile owns causal rolling feeds. ``batch`` binds
    preloaded research data and retains vectorized indicator precomputation;
    matching backends advance that graph through ``advance_bound`` and submit
    each callback through ``step_bound``. Both profiles own init/start/stop and
    business notification dispatch. Lite supports the batch profile only.
    """

    def __init__(
        self,
        strategy_cls=None,
        *,
        strategy=None,
        strategy_args=(),
        feeds=None,
        profile="stream",
        data_feeds=None,
        broker=None,
        parameters=None,
        capacity=4096,
        buffers=None,
        prefix_history_limit=100000,
        max_indicators=1024,
        max_plans=4096,
    ):
        from tradelearn.core.strategy_events import assert_observer_allows_mutation
        assert_observer_allows_mutation()
        self.profile = profile
        if profile == "batch":
            self._init_bound(strategy_cls, strategy_args, parameters, data_feeds, broker,
                             prefix_history_limit, max_indicators, max_plans)
            return
        if profile != "stream":
            raise ValueError("profile must be stream or batch")
        if (strategy_cls is None) == (strategy is None):
            raise ValueError("provide strategy_cls or strategy, exclusively")
        profile = strategy_cls if strategy_cls is not None else type(strategy)
        if not issubclass(profile, Strategy):
            raise TypeError("runtime requires TradeLearn Strategy")
        if any(
            base.__module__ == "tradelearn.lite.strategy" and base.__name__ == "Strategy"
            for base in profile.__mro__
        ):
            raise TypeError(
                "lite uses its batch research lifecycle; runtime supports engine/core strategies"
            )
        if not feeds or any(v not in ("bar", "tick") for v in feeds.values()):
            raise ValueError("explicit symbol -> bar/tick feeds required")
        if type(capacity) is not int or capacity < 2:
            raise ValueError("capacity must be at least two")
        if type(prefix_history_limit) is not int or prefix_history_limit < capacity:
            raise ValueError("prefix_history_limit must be an integer at least capacity")
        self.prefix_history_limit = prefix_history_limit
        if any(type(n) is not int or n < 1 for n in (max_indicators, max_plans)):
            raise ValueError("runtime graph budgets must be positive integers")
        self.max_indicators, self.max_plans = max_indicators, max_plans
        self._mutation_epoch = 0
        self._thread = get_ident()
        self._started = self._stopped = self._failed = False
        self._sequence = 0
        self._last_ts = {}
        self.context = None
        self._indicators = []
        self._plans = WeakValueDictionary()
        self._stepping = False
        self.modes = MappingProxyType(dict(feeds))
        self.feeds = {s: RuntimeFeed(self, s, capacity, (buffers or {}).get(s)) for s in feeds}
        datas = list(self.feeds.values())
        self.broker = broker
        self.strategy = strategy if strategy is not None else self._construct_strategy(
            strategy_cls, datas, broker, strategy_args, parameters
        )
        if not isinstance(self.strategy, Strategy):
            raise TypeError("runtime requires TradeLearn Strategy")
        if getattr(self.strategy, "_runtime", None) is not None:
            raise ValueError("strategy already belongs to a runtime")
        self.strategy._runtime = self
        self.strategy.datas = datas
        self.strategy.data = datas[0]
        self.strategy.runtime_context = None
        for i, feed in enumerate(datas):
            setattr(self.strategy, f"data{i}", feed)
        if broker is not None:
            self.strategy.broker = broker
        self.broker = self.strategy.broker
        for indicator in self.strategy._indicators:
            self._register_indicator(indicator)
        if callable(getattr(self.broker, "bind_runtime", None)):
            self.broker.bind_runtime(self)
        if callable(getattr(self.broker, "bind", None)):
            self.broker.bind(self.strategy)
        from tradelearn.core.strategy_events import assemble_observers
        assemble_observers(self.strategy)

    def add_observer(self, observer, **kwargs):
        self._owner()
        return self.strategy.add_observer(observer, **kwargs)

    def remove_observer(self, observer):
        self._owner()
        return self.strategy.remove_observer(observer)

    def invoke_component(self, callback, *args, **kwargs):
        """Coordinate legacy analyzer/value-sampling component callbacks."""
        self._owner()
        if not self._stopped:
            self._start_attempted = True
        try:
            with engine_context(data=self.strategy.data, datas=self.strategy.datas,
                                strategy=self.strategy):
                return callback(*args, **kwargs)
        except BaseException as error:
            self._failed = True
            self.cleanup(error)
            raise

    def _construct_strategy(self, strategy_cls, datas, broker, args, parameters):
        with engine_context(data=datas[0], datas=datas):
            if hasattr(strategy_cls, "_base_init"):
                return strategy_cls(*args, **(parameters or {}))
            instance = strategy_cls.__new__(strategy_cls)
            instance.datas, instance.data = datas, datas[0]
            instance._indicators = []
            instance.broker = broker
            for i, feed in enumerate(datas):
                setattr(instance, f"data{i}", feed)
            with engine_context(strategy=instance):
                strategy_cls.__init__(instance, *args, **(parameters or {}))
            return instance

    def _init_bound(self, strategy_cls, args, parameters, datas, broker,
                    prefix_limit, max_indicators, max_plans):
        """Preloaded storage backend for the shared incremental indicator graph."""
        self._thread = get_ident()
        self._started = self._stopped = self._failed = False
        self._initialized = self._stepping = False
        self._mutation_epoch = self._sequence = 0
        self.context = None
        self._indicators = []
        self._plans = WeakValueDictionary()
        self._compiled_lines = WeakValueDictionary()
        self.max_indicators, self.max_plans = max_indicators, max_plans
        self.prefix_history_limit = prefix_limit
        self.broker = broker
        self.feeds, self._bound_feeds = {}, {}
        for i, data in enumerate(datas):
            symbol = str(getattr(data, "_name", None) or i)
            feed = BoundFeed(self, symbol, data)
            self.feeds[symbol] = feed
            self._bound_feeds[id(data)] = feed
            data.runtime = self
            for name in feed._buffer.arrays:
                line = getattr(data, name, None)
                if line is not None:
                    line._runtime, line._bound_feed, line._runtime_field = self, feed, name
        self.modes = MappingProxyType({symbol: "bar" for symbol in self.feeds})
        self.strategy = self._construct_strategy(strategy_cls, datas, broker, args, parameters)
        self.strategy._runtime = self
        self.strategy.datas, self.strategy.data = datas, datas[0]
        self.strategy.broker = broker
        self.strategy.runtime_context = None
        self._bound_advancers = ()
        self._bound_data_advance = None
        self._bound_seen = {symbol: 0 for symbol in self.feeds}
        for indicator in tuple(self.strategy._indicators):
            self._register_indicator(indicator)
        from tradelearn.core.strategy_events import assemble_observers
        assemble_observers(self.strategy)

    def initialize(self):
        self._owner()
        if getattr(self, "_initialized", False):
            return
        self._start_attempted = True
        self._initialized = True
        if callable(getattr(self.strategy, "_setup", None)):
            self._invoke("_setup")
        self._invoke("init")

    def configure_bound(self, advancers, advance_datas=None):
        self._owner()
        self._bound_advancers = tuple(advancers)
        self._bound_data_advance = advance_datas
        install = getattr(self.strategy, "_set_bar_advancers", None)
        if callable(install):
            install(self._bound_advancers)

    def advance_bound(self, cursor, data_cursors=None):
        self._running()
        if data_cursors is not None:
            for data, position in zip(self.strategy.datas, data_cursors):
                data._advance(max(-1, position))
        elif self._bound_data_advance is not None:
            self._bound_data_advance(cursor)
        if callable(getattr(self.strategy, "_pre_next", None)):
            self._invoke("_pre_next", cursor)
        else:
            for advance in self._bound_advancers:
                advance(cursor)
        for index, feed in enumerate(self.feeds.values()):
            feed._buffer._cursor = data_cursors[index] if data_cursors is not None else cursor
        self._sequence += 1
        self._mutation_epoch += 1
        primary = next(iter(self.feeds.values()))
        clock_ts = primary.datetime[0]
        timestamp = int(clock_ts.value) if clock_ts is not None else 0
        active = []
        for symbol, feed in self.feeds.items():
            if feed.received <= self._bound_seen[symbol]:
                continue
            event = MappingProxyType(dict(
                symbol=symbol, ts=feed.datetime[0],
                **{name: float(getattr(feed, name)[0])
                   for name in ("open", "high", "low", "close", "volume")}
            ))
            active.append((symbol, feed, event))
        primary_context = self._market_context(
            [(symbol, event) for symbol, _, event in active], primary._name, timestamp
        )
        self._active_market_symbols = frozenset(symbol for symbol, _, _ in active)
        self._stepping = True
        try:
            for symbol, feed, event in active:
                feed.revision += 1
                self._bound_seen[symbol] = feed.received
                self.context = RuntimeContext(symbol, event, int(event["ts"].value), self._sequence)
                self.strategy.runtime_context = self.context
                for indicator in tuple(self._indicators):
                    self._update_indicator(indicator, symbol)
            self.context = primary_context
            self.strategy.runtime_context = primary_context
        except BaseException as error:
            self._failed = True
            self._stepping = False
            self.cleanup(error)
            raise

    def _warming_up(self):
        """One eligibility rule for every storage/clock profile."""
        minimum = int(getattr(self.strategy, "_manual_min_period", 0))
        if any(len(feed) < minimum for symbol, feed in self.feeds.items()
               if self.modes[symbol] == "bar"):
            return True
        explicit = getattr(type(self.strategy), "min_period", 0)
        explicit = explicit(self.strategy) if callable(explicit) else explicit
        primary = next(iter(self.feeds.values()))
        if int(explicit or 0) > 0 and len(primary) <= int(explicit):
            return True
        for indicator in self._indicators:
            feed = self._feed_for(indicator.data)
            if feed is not None and len(feed) < indicator.min_period:
                return True
        for line in tuple(self._plans.values()):
            if len(line.feed) < line.min_period:
                return True
        for line in getattr(self, "_compiled_lines", {}).values():
            feed = self._feed_for(line)
            if feed is not None and len(feed) < line.min_period:
                return True
        return False

    def bound_ready(self, cursor):
        self._running()
        return not self._warming_up()

    def step_bound(self, cursor, dispatch):
        """Dispatch one preloaded observation after backend matching.

        The backend's dispatch closure only manages order buffers around the selected callback.
        Warmup and callback choice belong to this coordinator.
        """
        self._running()
        ready = self.bound_ready(cursor)
        try:
            return ready, dispatch(cursor, "next" if ready else "prenext")
        finally:
            self._stepping = False

    def invoke_business(self, name, *args, **kwargs):
        self._owner()
        if (self._stopped or self._failed) and not getattr(self, "_stopping", False):
            raise RuntimeError("runtime is not accepting business callbacks")
        return self._invoke(name, *args, **kwargs)

    def _owner(self):
        from tradelearn.core.strategy_events import assert_observer_allows_mutation
        assert_observer_allows_mutation()
        if get_ident() != self._thread:
            raise RuntimeError("runtime events must use the owner thread")

    def _running(self):
        self._owner()
        if not self._started or self._stopped or self._failed:
            raise RuntimeError("runtime is not running")

    def _invoke(self, name, *args, **kwargs):
        try:
            with engine_context(
                data=self.strategy.data, datas=self.strategy.datas, strategy=self.strategy
            ):
                from tradelearn.core.strategy_events import invoke_callback
                return invoke_callback(self.strategy, name, *args, **kwargs)
        except BaseException as error:
            self._failed = True
            if name != "stop":
                self.cleanup(error)
            raise

    @property
    def accepts_new_orders(self):
        return not (self._failed or self._stopped or getattr(self, "_stopping", False))

    def assert_can_submit(self):
        self._owner()
        if not self.accepts_new_orders:
            raise RuntimeError("runtime is not accepting new orders")

    def cleanup(self, primary_error=None):
        """Stop once; preserve a primary failure and attach any stop failure."""
        self._owner()
        if primary_error is None:
            return self.stop()
        self._failed = True
        try:
            self.stop()
        except BaseException as cleanup_error:
            _safe_cleanup_diagnostic(primary_error, cleanup_error)

    def start(self):
        self._owner()
        if self._stopped or self._failed:
            raise RuntimeError("runtime cannot restart")
        if not self._started:
            self.initialize()
            self._started = True
            self._invoke("start")

    def stop(self):
        self._owner()
        if (self._started or getattr(self, "_start_attempted", False)) and not self._stopped:
            self._stopped = True
            self._stopping = True
            try:
                self._invoke("stop")
            finally:
                self._stopping = False
                from tradelearn.core.strategy_events import emit_domain
                emit_domain(self.strategy, "lifecycle.closed", "after", {"failed": self._failed})

    def prefix(self, function, *inputs, history="full", **kwargs):
        """Causal window-recomputed Series function; not a constant-time promise."""
        self._owner()
        line = next((x for x in inputs if isinstance(x, RuntimeLine)), None)
        if line is None:
            raise ValueError("prefix needs a runtime line input")
        return line._runtime_function(
            SimpleNamespace(_func=function, compute=function), inputs, kwargs, history=history
        )

    def _register_precomputed(self, line, minimum):
        self._owner()
        line.min_period = max(line.min_period, minimum)
        self._compiled_lines[id(line)] = line
        return line

    def causal_line(self, value):
        self._owner()
        if isinstance(value, RuntimeLine):
            return value
        feed = self._feed_for(value)
        if feed is None:
            raise ValueError("line does not belong to this runtime feed")
        field = getattr(value, "_runtime_field", None)
        if field is not None:
            return getattr(feed, field)
        cached = getattr(value, "_causal_runtime_line", None)
        if cached is not None and cached._runtime is self:
            return cached
        line = RuntimeLine(feed, compute=lambda: pd.Series(value._values[:feed.received]),
                           min_period=getattr(value, "min_period", 0), history="full")
        value._causal_runtime_line = line
        return line

    def output_line(self, data=None):
        self._owner()
        feed = self._feed_for(data)
        return RuntimeLine(feed if feed is not None else next(iter(self.feeds.values())))

    def _feed_for(self, value):
        if self.profile == "batch":
            bound = getattr(self, "_bound_feeds", {}).get(id(value))
            if bound is not None:
                return bound
            bound = getattr(value, "_bound_feed", None)
            if bound is not None:
                return bound
            source = getattr(value, "_cursor_source", None)
            bound = getattr(self, "_bound_feeds", {}).get(id(source))
            if bound is not None:
                return bound
        if isinstance(value, RuntimeFeed):
            return value
        if isinstance(value, RuntimeLine):
            return value.feed
        if hasattr(value, "lines"):
            try:
                return self._feed_for(value.lines[0])
            except (IndexError, TypeError):
                pass
        return None

    def normalize_indicator(self, indicator):
        self._owner()
        for name in indicator.lines_def:
            value = getattr(indicator.lines, name)
            if hasattr(value, "lines"):
                value = value.lines[0]
                setattr(indicator.lines, name, value)
            if isinstance(value, (int, float, np.number)):
                feed = self._feed_for(indicator.data)
                if feed is None:
                    feed = next(iter(self.feeds.values()))
                line = RuntimeLine(feed)
                line.compute = lambda v=float(value), f=feed, line=line: pd.Series(
                    v, index=range(f.received if line.history == "full" else len(f))
                )
                if len(self._plans) >= self.max_plans:
                    raise RuntimeError("runtime graph plan limit reached")
                self._plans[id(line)] = line
                setattr(indicator.lines, name, line)

    @property
    def failed(self):
        return self._failed

    def _register_indicator(self, indicator):
        self._owner()
        if not hasattr(indicator, "lines_def"):
            return  # Legacy LineRoot analyzers are components, not indicator graphs.
        if any(x is indicator for x in self._indicators):
            return
        if len(self._indicators) >= self.max_indicators:
            raise RuntimeError("runtime indicator limit reached")
        self.normalize_indicator(indicator)
        lines = [getattr(indicator.lines, n) for n in indicator.lines_def]
        if any(not isinstance(line, RuntimeLine) and not (
                self.profile == "batch" and isinstance(line, LineSeries)) for line in lines):
            raise ValueError("indicator must expose runtime lines or a prefix function plan")
        self._indicators.append(indicator)
        if self.context is not None:
            feed = self._feed_for(indicator.data)
            symbol = (feed._name if self._stepping and feed is not None
                      and feed._name in self._active_market_symbols else self.context.symbol)
            self._update_indicator(indicator, symbol)

    def _update_indicator(self, indicator, symbol):
        feed = self._feed_for(getattr(indicator, "data", None))
        if feed is None:
            feed = self.strategy.data
        if feed is not self.feeds[symbol] or self.modes[symbol] != "bar":
            return
        if getattr(indicator, "_runtime_sequence", None) == self._sequence:
            return
        indicator._runtime_sequence = self._sequence
        callback = getattr(indicator, "next", None)
        if callable(callback):
            for name in indicator.lines_def:
                line = getattr(indicator.lines, name)
                if (isinstance(line, RuntimeLine) and line.compute is None and line.field is None
                        and getattr(line, "_append_sequence", None) != self._sequence):
                    line._append_sequence = self._sequence
                    line._append_output()
            callback = (
                callback
                if len(feed) >= indicator.min_period
                else getattr(indicator, "prenext", lambda: None)
            )
            with engine_context(
                data=self.strategy.data, datas=self.strategy.datas, strategy=self.strategy
            ):
                callback()
        else:
            for name in indicator.lines_def:
                line = getattr(indicator.lines, name)
                if isinstance(line, RuntimeLine):
                    line.to_series()

    def _prepare_market(self, symbol, event, kind):
        if self.profile != "stream":
            raise ValueError("market ingestion requires the stream profile")
        if self.modes.get(symbol) != kind:
            raise ValueError("event does not match configured feed mode")
        feed = self.feeds[symbol]
        if kind == "tick":
            timestamp = int(event.ts_event)
            if timestamp < 0:
                raise ValueError("negative tick timestamp")
            return symbol, event, kind, timestamp, None, event
        values = dict(event) if isinstance(event, Mapping) else {
            name: getattr(event, name) for name in ("ts", "open", "high", "low", "close", "volume")
        }
        ts = pd.Timestamp(values["ts"])
        if ts.tzinfo is None:
            raise ValueError("bar timestamp must be timezone aware")
        timestamp = int(ts.value)
        numbers = {name: float(values[name]) for name in ("open", "high", "low", "close", "volume")}
        if not all(math.isfinite(v) for v in numbers.values()) or numbers["volume"] < 0:
            raise ValueError("invalid bar numbers")
        if not numbers["low"] <= min(numbers["open"], numbers["close"]) <= max(
            numbers["open"], numbers["close"]
        ) <= numbers["high"]:
            raise ValueError("invalid OHLC bounds")
        if timestamp < self._last_ts.get(symbol, -1):
            raise ValueError("bar timestamps must be monotonic per symbol")
        if feed._history is not None and feed.received >= self.prefix_history_limit:
            self._failed = True
            raise RuntimeError("full-prefix history limit reached; runtime stopped before accepting event")
        normalized = MappingProxyType(dict(symbol=symbol, ts=ts, **numbers))
        return symbol, event, kind, timestamp, dict(numbers, datetime=ts.timestamp()), normalized

    def _advance_market(self, prepared):
        symbol, event, kind, timestamp, record, _ = prepared
        feed = self.feeds[symbol]
        if kind == "bar":
            feed._buffer.append(record)
            feed.revision += 1
            self._mutation_epoch += 1
            feed.received += 1
            if feed._history is not None:
                for name, value in record.items():
                    feed._history[name].append(value)
        else:
            feed.quote = event
        self._last_ts[symbol] = max(timestamp, self._last_ts.get(symbol, -1))

    def _market_context(self, active, clock_symbol, timestamp):
        event = active[0][1] if len(active) == 1 else MappingProxyType(dict(
            clock_symbol=clock_symbol, ts=pd.Timestamp(timestamp, unit="ns", tz="UTC"),
            active_symbols=tuple(symbol for symbol, _ in active), bars=tuple(active),
        ))
        return RuntimeContext(clock_symbol, event, timestamp, self._sequence)

    def _dispatch_market(self, prepared):
        # Every explicitly grouped window is visible before any indicator runs.
        self._sequence += 1
        active = [(item[0], item[5]) for item in prepared]
        context = self._market_context(active, prepared[0][0], prepared[0][3])
        self._active_market_symbols = frozenset(symbol for symbol, _ in active)
        self._stepping = True
        try:
            for symbol, _, _, timestamp, _, event in prepared:
                self.context = RuntimeContext(symbol, event, timestamp, self._sequence)
                self.strategy.runtime_context = self.context
                for indicator in tuple(self._indicators):
                    self._update_indicator(indicator, symbol)
            self.context = context
            self.strategy.runtime_context = context
            self._invoke("prenext" if self._warming_up() else "next")
        finally:
            self._stepping = False
        return context

    def step_market(self, symbol, event, kind="bar"):
        """Accept one arrival and dispatch once; never waits for other symbols."""
        self._running()
        try:
            prepared = self._prepare_market(symbol, event, kind)
            self._advance_market(prepared)
            return self._dispatch_market((prepared,))
        except BaseException as error:
            self._failed = True
            self.cleanup(error)
            raise

    def step_market_batch(self, events):
        """Accept an explicit same-timestamp bar group and dispatch exactly once.

        Items are ``(symbol, bar)`` pairs or symbol-bearing StreamBar objects.
        No hidden buffering, alignment, execution report delivery or loop runs.
        Validation rejects empty groups, duplicate symbols and mixed source times
        before changing a feed. Input order supplies the explicit clock symbol.
        """
        self._running()
        prepared, symbols = [], set()
        for item in events:
            if isinstance(item, tuple) and len(item) == 2:
                symbol, event = item
            else:
                symbol = item.get("symbol") if isinstance(item, Mapping) else getattr(item, "symbol", None)
                event = item
            if symbol in symbols:
                raise ValueError("market group requires distinct symbols")
            symbols.add(symbol)
            try:
                value = self._prepare_market(symbol, event, "bar")
            except BaseException as error:
                self._failed = True
                self.cleanup(error)
                raise
            if prepared and value[3] != prepared[0][3]:
                raise ValueError("market group requires one source timestamp")
            prepared.append(value)
        if not prepared:
            raise ValueError("market group cannot be empty")
        try:
            for value in prepared:
                self._advance_market(value)
            return self._dispatch_market(prepared)
        except BaseException as error:
            self._failed = True
            self.cleanup(error)
            raise

    def step_broker(self, event, kind="order", owner=None):
        self._owner()
        if self._stopped or self._failed:
            raise RuntimeError("runtime is not accepting broker events")
        if owner is not None and owner is not self.strategy:
            raise ValueError("broker event belongs to a different strategy")
        from tradelearn.core.broker_events import BrokerEvent

        if isinstance(event, BrokerEvent):
            if event.payload is None:
                return
            event = event.payload
        if kind not in ("order", "trade"):
            raise ValueError("broker notification kind must be order or trade")
        return self._invoke("notify_" + kind, event)


__all__ = ["StrategyRuntime", "RuntimeContext"]
