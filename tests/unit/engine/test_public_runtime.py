from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tradelearn.backtest.event_runtime import StrategyRuntime
from tradelearn.engine import Indicator, Strategy
from tradelearn.indicators.base import FunctionIndicator


def bar(n, value):
    return dict(
        ts=pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(n, "s"),
        open=value,
        high=value,
        low=value,
        close=value,
        volume=1,
    )


def test_lifecycle_multifeed_history_and_tick_prefix():
    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.seen = []
            self.lifecycle = []

        def start(self):
            self.lifecycle.append("start")

        def next(self):
            self.seen.append((self.runtime_context.symbol, len(self.data0), len(self.data1)))

        def notify_order(self, o):
            self.lifecycle.append(o)

        def stop(self):
            self.lifecycle.append("stop")

    r = StrategyRuntime(S, feeds={"A": "bar", "B": "tick"}, capacity=2)
    r.start()
    r.start()
    r.step_broker("order")
    r.step_market("A", bar(1, 2))
    r.step_market("B", SimpleNamespace(ts_event=2), kind="tick")
    assert r.strategy.seen == [("A", 1, 0), ("B", 1, 0)]
    assert r.feeds["B"].quote.ts_event == 2
    r.step_market("A", bar(2, 3))
    r.step_market("A", bar(3, 4))
    assert r.strategy.history_panel().close.tolist() == [3, 4]
    assert r.feeds["A"].close.to_series().tolist() == [3, 4]
    with pytest.raises(ValueError):
        r.step_market("A", bar(0, 1))
    r.stop()
    r.stop()
    assert r.strategy.lifecycle == ["start", "order", "stop"]
    with pytest.raises(RuntimeError):
        r.step_market("A", bar(4, 5))


def test_real_indicator_init_start_next_nested_and_warmup():
    sma = FunctionIndicator("sma", lambda s, period=2: s.rolling(period).mean())

    class Mean(Indicator):
        lines = ("mean",)
        params = (("period", 2),)

        def __init__(self):
            self.lines.mean = sma(self.data.close, period=self.p.period)

    class Nested(Indicator):
        lines = ("twice",)

        def __init__(self):
            self.inner = Mean(self.data)
            self.lines.twice = self.inner.mean * 2

    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.i = Nested(self.data)
            self.seen = []
            self.pre = 0

        def start(self):
            self.started = Mean(self.data)

        def prenext(self):
            self.pre += 1

        def next(self):
            dynamic = Mean(self.data)
            self.seen.append((self.i[0], self.started[-1], dynamic[0]))

    r = StrategyRuntime(S, feeds={"A": "bar"})
    r.start()
    for i, x in enumerate([2, 4, 6]):
        r.step_market("A", bar(i, x))
    assert r.strategy.pre == 1
    assert r.strategy.seen[0][0] == 6
    assert np.isnan(r.strategy.seen[0][1])
    assert r.strategy.seen[1] == (10, 3, 5)


def test_incremental_indicator_next_once_before_strategy():
    class Counter(Indicator):
        lines = ("value",)

        def __init__(self):
            self.calls = 0

        def next(self):
            self.calls += 1
            self.lines.value[0] = self.data.close[0] * 2

    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.i = Counter(self.data)
            self.values = []

        def next(self):
            self.values.append((self.i[0], self.i[-1], self.i.calls))

    r = StrategyRuntime(S, feeds={"A": "bar", "B": "bar"})
    r.start()
    r.step_market("A", bar(0, 3))
    r.step_market("B", bar(0, 8))
    r.step_market("A", bar(1, 5))
    assert r.strategy.values[0][0] == 6
    assert r.strategy.values[-1] == (10, 6, 2)


def test_unsupported_static_indicator_is_rejected_not_frozen():
    from tradelearn.backtest.lines import LineSeries

    class Frozen(Indicator):
        lines = ("v",)

        def __init__(self):
            self.lines.v = LineSeries([1, 2, 3])

    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.f = Frozen(self.data)

    with pytest.raises(ValueError, match="runtime line|prefix"):
        StrategyRuntime(S, feeds={"A": "bar"})


def test_actual_pta_indicator_warms_on_prefix_and_supports_constant_lines():
    import tradelearn.engine as bt

    class Mean(Indicator):
        lines = ("sma", "constant")

        def __init__(self):
            self.lines.sma = bt.pta.SMA(self.data.close, length=2)
            self.lines.constant = 7

    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.i = Mean(self.data)
            self.values = []

        def next(self):
            self.values.append((self.i.sma[0], self.i.constant[0]))

    r = StrategyRuntime(S, feeds={"A": "bar"})
    r.start()
    r.step_market("A", bar(1, 4))
    r.step_market("A", bar(2, 6))
    assert r.strategy.values == [(5, 7)]


def test_prefix_plan_rejects_future_and_clock_ownership_and_fails_closed():
    from concurrent.futures import ThreadPoolExecutor

    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.values = []

        def next(self):
            self.values.append(self.data.close.to_series().tolist())

    r = StrategyRuntime(S, feeds={"A": "bar"})
    r.start()
    r.step_market("A", bar(1, 2))
    with pytest.raises(ValueError, match="future"):
        r.feeds["A"].close[1]
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError, match="thread"):
            pool.submit(r.step_market, "A", bar(2, 4)).result()
    assert r.strategy.values == [[2]]

    class Bad(S):
        def notify_order(self, event):
            raise LookupError("bad")

    x = StrategyRuntime(Bad, feeds={"A": "bar"})
    x.start()
    with pytest.raises(LookupError):
        x.step_broker(object())
    with pytest.raises(RuntimeError):
        x.step_market("A", bar(1, 2))


def test_direct_function_plan_updates_and_warms_without_indicator_wrapper():
    sma = FunctionIndicator("sma", lambda s, period=2: s.rolling(period).mean())

    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.line = sma(self.data.close, period=2)
            self.values = []

        def next(self):
            self.values.append(self.line[0])

    r = StrategyRuntime(S, feeds={"A": "bar"})
    r.start()
    for n, v in enumerate([2, 4, 6]):
        r.step_market("A", bar(n, v))
    assert r.strategy.values == [3, 5]


def test_runtime_prefix_never_receives_future_and_retains_bounded_window():
    calls = []

    class S(Strategy):
        def __init__(self):
            super().__init__()

            def compute(s):
                calls.append(s.tolist())
                return s * 2

            self.line = self.data.runtime.prefix(compute, self.data.close, history="window")
            self.values = []

        def next(self):
            self.values.append((self.line[0], self.line[0]))

    r = StrategyRuntime(S, feeds={"A": "bar"}, capacity=2)
    r.start()
    for n, v in enumerate([1, 2, 3]):
        r.step_market("A", bar(n, v))
    assert calls == [[1], [1, 2], [2, 3]]
    assert r.strategy.values == [(2, 2), (4, 4), (6, 6)]


def test_ema_keeps_full_causal_prefix_across_window_rollover():
    ema = FunctionIndicator("ema", lambda s, span=2: s.ewm(span=span, adjust=False).mean())

    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.i = ema(self.data.close)
            self.values = []

        def next(self):
            self.values.append(self.i[0])

    r = StrategyRuntime(S, feeds={"A": "bar"}, capacity=2)
    r.start()
    for n, v in enumerate([1, 2, 3, 4]):
        r.step_market("A", bar(n, v))
    assert r.strategy.values == pytest.approx(
        pd.Series([1, 2, 3, 4]).ewm(span=2, adjust=False).mean().tolist()
    )
    assert len(r.feeds["A"].close.to_series()) == 2


def test_frozen_array_math_and_function_args_fail_at_construction():
    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.line = self.data.close + np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="scalar|tracked"):
        StrategyRuntime(S, feeds={"A": "bar"})
    func = FunctionIndicator("f", lambda s, other: s + other)

    class T(Strategy):
        def __init__(self):
            super().__init__()
            self.line = func(self.data.close, np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="scalar|tracked"):
        StrategyRuntime(T, feeds={"A": "bar"})


def test_nested_incremental_indicators_and_late_output_alignment():
    class Inner(Indicator):
        lines = ("v",)

        def next(self):
            self.lines.v[0] = self.data.close[0] * 2

    class Outer(Indicator):
        lines = ("v",)

        def next(self):
            self.lines.v[0] = self.data[0] + 1

    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.i = Outer(Inner(self.data))
            self.values = []
            self.late = None

        def next(self):
            if self.late is None and self.data.close[0] == 3:
                self.late = Inner(self.data)
            self.values.append(
                (self.i[0], None if self.late is None else (self.late[0], len(self.late)))
            )

    r = StrategyRuntime(S, feeds={"A": "bar"}, capacity=3)
    r.start()
    for n, v in enumerate([1, 2, 3, 4]):
        r.step_market("A", bar(n, v))
    assert r.strategy.values == [(3, None), (5, None), (7, (6, 3)), (9, (8, 3))]


def test_full_prefix_limit_stops_before_accepting_market_step():
    ema = FunctionIndicator("ema", lambda s: s.ewm(span=2, adjust=False).mean())

    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.i = ema(self.data.close)

    r = StrategyRuntime(S, feeds={"A": "bar"}, capacity=2, prefix_history_limit=3)
    r.start()
    for n, v in enumerate([1, 2, 3]):
        r.step_market("A", bar(n, v))
    with pytest.raises(RuntimeError, match="history limit"):
        r.step_market("A", bar(4, 4))
    assert r.feeds["A"].close[0] == 3
    with pytest.raises(RuntimeError):
        r.step_market("A", bar(5, 5))


def test_cross_feed_functions_do_not_positionally_align_independent_clocks():
    f = FunctionIndicator("spread", lambda a, b: a - b)

    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.line = f(self.data0.close, self.data1.close)

    with pytest.raises(ValueError, match="alignment"):
        StrategyRuntime(S, feeds={"A": "bar", "B": "bar"})


def test_incremental_indicator_explicit_minperiod_before_next():
    class C(Indicator):
        lines = ("v",)

        def __init__(self):
            self.addminperiod(2)
            self.calls = 0

        def next(self):
            self.calls += 1
            self.v[0] = self.data.close[0]

    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.i = C(self.data)
            self.values = []

        def next(self):
            self.values.append(self.i[0])

    r = StrategyRuntime(S, feeds={"A": "bar"})
    r.start()
    r.step_market("A", bar(1, 2))
    r.step_market("A", bar(2, 4))
    assert r.strategy.values == [4]
    assert r.strategy.i.calls == 1


def test_nested_sma_minimum_period_composes_and_keyword_line_is_tracked():
    import tradelearn.engine as bt

    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.inner = bt.pta.SMA(close=self.data.close, length=3)
            self.outer = bt.pta.SMA(close=self.inner, length=3)
            self.values = []

        def next(self):
            self.values.append(self.outer[0])

    r = StrategyRuntime(S, feeds={"A": "bar"})
    r.start()
    for n, v in enumerate([1, 2, 3, 4, 5, 6]):
        r.step_market("A", bar(n, v))
    assert r.strategy.values == [3, 4]


def test_lite_research_profile_rejected_before_init():
    from tradelearn.lite import Strategy as LiteStrategy

    class S(LiteStrategy):
        def init(self):
            raise AssertionError("research lifecycle must not be partly run")

    with pytest.raises(TypeError, match="lite"):
        StrategyRuntime(S, feeds={"A": "bar"})


def test_output_write_invalidates_same_step_dependency_cache():
    class Incremental(Indicator):
        lines = ("v",)

        def __init__(self):
            self.twice = self.v * 2

        def next(self):
            self.twice[0]
            self.v[0] = self.data.close[0]

    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.i = Incremental(self.data)
            self.values = []

        def next(self):
            self.values.append((self.i[0], self.i.twice[0]))

    r = StrategyRuntime(S, feeds={"A": "bar"})
    r.start()
    for n, v in enumerate([1, 2]):
        r.step_market("A", bar(n, v))
    assert r.strategy.values == [(1, 2), (2, 4)]


def test_constant_indicator_can_feed_full_prefix_at_construction_and_rollover():
    sma = FunctionIndicator("sma", lambda s, period=2: s.rolling(period).mean())

    class C(Indicator):
        lines = ("v",)

        def __init__(self):
            self.lines.v = 7

    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.i = sma(C(self.data).v, period=2)
            self.values = []

        def next(self):
            self.values.append(self.i[0])

    r = StrategyRuntime(S, feeds={"A": "bar"}, capacity=2)
    r.start()
    for n in range(4):
        r.step_market("A", bar(n, 1))
    assert r.strategy.values == [7, 7, 7]


def test_invalid_plan_does_not_enable_input_history():
    r = StrategyRuntime(Strategy, feeds={"A": "bar"}, capacity=2)
    with pytest.raises(ValueError):
        r.prefix(lambda a, b: a + b, r.feeds["A"].close, np.array([1, 2]))
    assert r.feeds["A"]._history is None


def test_dynamic_indicator_budget_fails_closed():
    class Incremental(Indicator):
        lines = ("v",)

        def next(self):
            self.v[0] = 1

    class S(Strategy):
        def next(self):
            Incremental(self.data)

    r = StrategyRuntime(S, feeds={"A": "bar"}, max_indicators=2)
    r.start()
    r.step_market("A", bar(1, 1))
    r.step_market("A", bar(2, 2))
    with pytest.raises(RuntimeError, match="indicator limit"):
        r.step_market("A", bar(3, 3))
    with pytest.raises(RuntimeError):
        r.step_market("A", bar(4, 4))


def test_shared_dag_version_work_is_linear_in_unique_edges(monkeypatch):
    from tradelearn.backtest.event_runtime import RuntimeLine

    r = StrategyRuntime(Strategy, feeds={"A": "bar"})
    line = r.feeds["A"].close
    for _ in range(10):
        line = line + line
    original = RuntimeLine.version.fget
    calls = []

    def version(self):
        calls.append(id(self))
        return original(self)

    monkeypatch.setattr(RuntimeLine, "version", property(version))
    _ = line.version
    assert len(calls) <= 25


def test_constant_line_input_preserves_secondary_feed_clock():
    class Constant(Indicator):
        lines = ("v",)

        def __init__(self):
            self.lines.v = 7

    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.c = Constant(self.data1.close)

    r = StrategyRuntime(S, feeds={"A": "bar", "B": "bar"})
    r.start()
    r.step_market("B", bar(1, 2))
    assert r.strategy.c.v.feed is r.feeds["B"]
    assert r.strategy.c[0] == 7


def test_core_strategy_constructor_receives_bound_data_without_facade_metaclass():
    from tradelearn.backtest.strategy import Strategy as CoreStrategy

    class Core(CoreStrategy):
        def __init__(self, threshold):
            super().__init__()
            self.bound = self.data._name
            self.threshold = threshold

    r = StrategyRuntime(Core, feeds={"A": "bar"}, strategy_args=(3,))
    assert r.strategy.bound == "A"
    assert r.strategy.threshold == 3
