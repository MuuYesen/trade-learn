"""The Strategy clock follows all grouped inputs, not only Data0."""

import backtrader as official
import pandas as pd
import pytest

from tradelearn import engine as tl
from tradelearn.backtest.event_runtime import StrategyRuntime


def run(api, stream=False):
    rows = []

    def frame(days):
        return pd.DataFrame(
            {k: [10.0] * len(days) for k in ("open", "high", "low", "close", "volume")},
            index=pd.to_datetime(["2026-01-" + d for d in days]),
        )

    class Business(api.Strategy):
        def record(self):
            rows.append(
                (
                    self.datetime[0],
                    self.datetime.datetime(),
                    list(self.datetime.get(size=1)),
                    self.datetime[-1] if len(self) > 1 else None,
                    self.data0.datetime.datetime(),
                )
            )

        prenext = nextstart = next = record

    first, second = frame(["01", "04"]), frame(["02", "03", "04"])
    if stream:
        runtime = StrategyRuntime(Business, feeds={"A": "bar", "B": "bar"}, capacity=2)
        try:
            runtime.start()
            for stamp in first.index.union(second.index):
                group = [
                    (symbol, dict(ts=stamp.tz_localize("UTC"), **data.loc[stamp].to_dict()))
                    for symbol, data in [("A", first), ("B", second)]
                    if stamp in data.index
                ]
                runtime.step_market_batch(group)
            runtime.stop()
        finally:
            runtime.cleanup()
    else:
        engine = api.Cerebro(stdstats=False)
        for data in [first, second]:
            engine.adddata(api.feeds.PandasData(dataname=data))
        engine.addstrategy(Business)
        engine.run()
    return rows


def test_stream_strategy_datetime_and_lag_match_fixed_bt_with_stale_primary_data():
    expected = run(official)
    actual = run(tl, stream=True)
    assert len(actual) == len(expected) == 4
    for value, want in zip(actual, expected, strict=True):
        assert value[0] == pytest.approx(want[0])
        assert value[1] == want[1]
        assert value[2] == pytest.approx(want[2])
        assert value[3] == want[3]
        assert value[4] == want[4]


def test_master_clock_does_not_replace_late_source_event_ns_or_register_fake_data():
    rows = []

    class Business(tl.Strategy):
        def record(self):
            rows.append((self.datetime.datetime(), self._runtime.context.ts_event_ns))

        prenext = nextstart = next = record

    runtime = StrategyRuntime(Business, feeds={"A": "bar", "B": "bar"}, capacity=2)
    try:
        runtime.start()
        for symbol, day in [("A", "01"), ("A", "05"), ("B", "03")]:
            stamp = pd.Timestamp("2026-01-" + day, tz="UTC")
            runtime.step_market(
                symbol, dict(ts=stamp, open=10, high=10, low=10, close=10, volume=1)
            )
        assert rows[-1] == (
            pd.Timestamp("2026-01-05").to_pydatetime(),
            pd.Timestamp("2026-01-03", tz="UTC").value,
        )
        assert set(runtime.feeds) == set(runtime.modes) == {"A", "B"}
        assert len(runtime.strategy.datas) == 2
        assert runtime._clock_feed.received == 3 and runtime._clock_feed.retained == 2
        assert set(runtime._clock_feed._buffer.arrays) == {"datetime"}
        assert runtime._clock_feed._history is None
        assert len(runtime._plans) == 0
    finally:
        runtime.cleanup()


def test_explicit_full_clock_history_limit_rejects_before_accepting_another_data_bar():
    class Business(tl.Strategy):
        def __init__(self):
            self.datetime.require_full()

    runtime = StrategyRuntime(Business, feeds={"A": "bar"}, capacity=2, prefix_history_limit=2)
    try:
        runtime.start()
        for day in ["01", "02"]:
            runtime.step_market(
                "A",
                dict(
                    ts="2026-01-" + day + "T00:00:00Z", open=10, high=10, low=10, close=10, volume=1
                ),
            )
        assert runtime.feeds["A"].received == runtime._clock_feed.received == 2
        with pytest.raises(RuntimeError, match="full-prefix clock history limit"):
            runtime.step_market(
                "A", dict(ts="2026-01-03T00:00:00Z", open=10, high=10, low=10, close=10, volume=1)
            )
        assert runtime.feeds["A"].received == runtime._clock_feed.received == 2
        assert runtime.failed
    finally:
        runtime.cleanup()
