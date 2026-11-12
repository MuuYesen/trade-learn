"""BT user datetime view on both storage paths; raw event clocks stay Unix/ns."""

from datetime import timezone

import backtrader as official
import pandas as pd
import pytest

from tradelearn import engine as bt
from tradelearn.backtest.event_runtime import StrategyRuntime

STAMPS = ["2026-01-01T09:00:00Z", "2026-01-01T09:01:00Z"]


def snapshot(data):
    line = data.datetime
    return (
        line[0],
        line.datetime(),
        line.date(),
        line.time(),
        line.datetime(tz=timezone.utc, naive=False),
        list(line.get(size=1)),
    )


def official_calls():
    calls = []

    class Business(official.Strategy):
        def next(self):
            calls.append(snapshot(self.data))

    e = official.Cerebro(stdstats=False)
    frame = pd.DataFrame(
        {name: [10.0, 11.0] for name in ("open", "high", "low", "close", "volume")},
        index=pd.to_datetime(STAMPS),
    )
    e.adddata(official.feeds.PandasData(dataname=frame))
    e.addstrategy(Business)
    e.run()
    return calls


def test_engine_stream_datetime_units_methods_and_timezone_match_actual_bt():
    calls = []

    class Business(bt.Strategy):
        def next(self):
            calls.append(snapshot(self.data))

    runtime = StrategyRuntime(Business, feeds={"A": "bar"})
    try:
        runtime.start()
        for stamp, value in zip(STAMPS, [10.0, 11.0], strict=True):
            runtime.step_market(
                "A", dict(ts=stamp, open=value, high=value, low=value, close=value, volume=1)
            )
        runtime.stop()
    finally:
        runtime.cleanup()
    expected = official_calls()
    assert len(calls) == len(expected)
    for actual, wanted in zip(calls, expected, strict=True):
        assert actual[0] == pytest.approx(wanted[0])
        assert actual[1:5] == wanted[1:5]
        assert actual[5] == pytest.approx(wanted[5])


def test_analyzer_bar_uses_actual_source_timestamp_instead_of_cursor_count():
    from tradelearn.backtest.engine import _current_bar

    runtime = StrategyRuntime(bt.Strategy, feeds={"A": "bar"})
    try:
        runtime.start()
        runtime.step_market("A", dict(ts=STAMPS[0], open=10, high=10, low=10, close=10, volume=1))
        assert _current_bar(runtime.strategy.data).datetime == pd.Timestamp(STAMPS[0])
    finally:
        runtime.cleanup()


def test_stream_feed_line_aliases_and_unknown_openinterest_match_bt_pandas_feed():
    calls = []

    class Business(bt.Strategy):
        def next(self):
            data = self.data
            calls.append(
                (
                    data.lines.getlinealiases(),
                    data.lines[0] is data.close,
                    data.lines[6] is data.datetime,
                    pd.isna(data.openinterest[0]),
                )
            )

    runtime = StrategyRuntime(Business, feeds={"A": "bar"})
    try:
        runtime.start()
        runtime.step_market("A", dict(ts=STAMPS[0], open=10, high=10, low=10, close=10, volume=1))
    finally:
        runtime.cleanup()
    assert calls == [
        (("close", "low", "high", "open", "volume", "openinterest", "datetime"), True, True, True)
    ]


def test_datetime_facade_keeps_raw_clock_and_plain_lite_datetime_convenience():
    from tradelearn.lite import Strategy as LiteStrategy

    for strategy in [bt.Strategy, LiteStrategy]:
        runtime = StrategyRuntime(strategy, feeds={"A": "bar"})
        try:
            runtime.start()
            runtime.step_market(
                "A", dict(ts=STAMPS[0], open=10, high=10, low=10, close=10, volume=1)
            )
            assert runtime.context.ts_event_ns == pd.Timestamp(STAMPS[0]).value
            assert (
                runtime.feeds["A"]._buffer.value("datetime", ago=0)
                == pd.Timestamp(STAMPS[0]).timestamp()
            )
            if strategy is LiteStrategy:
                assert runtime.strategy.data.datetime[0] == pd.Timestamp(STAMPS[0])
        finally:
            runtime.cleanup()


def test_missing_openinterest_does_not_consume_the_user_graph_plan_budget():
    class Business(bt.Strategy):
        def __init__(self):
            self.shifted = self.data.close + 1

    runtime = StrategyRuntime(Business, feeds={"A": "bar"}, max_plans=1)
    try:
        runtime.start()
        runtime.step_market("A", dict(ts=STAMPS[0], open=10, high=10, low=10, close=10, volume=1))
        assert runtime.strategy.shifted[0] == 11
        assert pd.isna(runtime.strategy.data.openinterest[0])
    finally:
        runtime.cleanup()


def test_engine_date_and_time_timezone_kwargs_match_bt():
    from datetime import timedelta

    expected = []
    actual = []
    west = timezone(timedelta(hours=-10))
    east = timezone(timedelta(hours=9))

    def read(data):
        return data.datetime.date(tz=west), data.datetime.time(tz=east, naive=False)

    class Reference(official.Strategy):
        def next(self):
            expected.append(read(self.data))

    frame = pd.DataFrame(
        {name: [10.0] for name in ("open", "high", "low", "close", "volume")},
        index=pd.to_datetime(STAMPS[:1]),
    )
    e = official.Cerebro(stdstats=False)
    e.adddata(official.feeds.PandasData(dataname=frame))
    e.addstrategy(Reference)
    e.run()

    class Business(bt.Strategy):
        def next(self):
            actual.append(read(self.data))

    runtime = StrategyRuntime(Business, feeds={"A": "bar"})
    try:
        runtime.start()
        runtime.step_market("A", dict(ts=STAMPS[0], open=10, high=10, low=10, close=10, volume=1))
    finally:
        runtime.cleanup()
    assert actual == expected


def test_preloaded_optional_openinterest_is_prefix_limited_during_initialization():
    observations = []

    class Business(bt.Strategy):
        def __init__(self):
            observations.append(("init", self.data.openinterest.to_series().tolist()))

        def next(self):
            observations.append(("next", self.data.openinterest.to_series().tolist()))

    frame = pd.DataFrame(
        {name: [10.0, 11.0] for name in ("open", "high", "low", "close", "volume")},
        index=pd.to_datetime(STAMPS),
    )
    frame["openinterest"] = [7.0, 9.0]
    e = bt.Cerebro(stdstats=False)
    e.adddata(bt.feeds.PandasData(dataname=frame))
    e.addstrategy(Business)
    e.run()
    assert observations == [("init", []), ("next", [7.0]), ("next", [7.0, 9.0])]


def test_stream_indicator_default_data_line_is_close_like_bt():
    from tradelearn.indicators import talib

    actual = []
    expected = []

    class Reference(official.Strategy):
        def __init__(self):
            self.mean = official.ind.SMA(self.data, period=2)

        def next(self):
            expected.append(self.mean[0])

    class Business(bt.Strategy):
        def __init__(self):
            self.mean = talib.SMA(self.data, timeperiod=2)

        def next(self):
            actual.append(self.mean[0])

    frame = pd.DataFrame(
        {name: [10.0, 11.0] for name in ("open", "high", "low", "close", "volume")},
        index=pd.to_datetime(STAMPS),
    )
    engine = official.Cerebro(stdstats=False)
    engine.adddata(official.feeds.PandasData(dataname=frame))
    engine.addstrategy(Reference)
    engine.run()
    runtime = StrategyRuntime(Business, feeds={"A": "bar"})
    try:
        runtime.start()
        for stamp, value in zip(STAMPS, [10.0, 11.0], strict=True):
            runtime.step_market(
                "A", dict(ts=stamp, open=value, high=value, low=value, close=value, volume=1)
            )
    finally:
        runtime.cleanup()
    assert actual == pytest.approx(expected)


def test_analyzer_bar_retains_exact_source_ns_and_each_feed_own_timestamp():
    from tradelearn.backtest.engine import _current_bar

    runtime = StrategyRuntime(bt.Strategy, feeds={"A": "bar", "B": "bar"})
    first = pd.Timestamp("2026-01-01T09:00:00.123456789Z")
    second = first + pd.Timedelta(seconds=1)
    try:
        runtime.start()
        for symbol, stamp in [("A", first), ("B", second)]:
            runtime.step_market(
                symbol, dict(ts=stamp, open=10, high=10, low=10, close=10, volume=1)
            )
        assert _current_bar(runtime.feeds["A"]).datetime == first
        assert _current_bar(runtime.feeds["B"]).datetime == second
    finally:
        runtime.cleanup()
