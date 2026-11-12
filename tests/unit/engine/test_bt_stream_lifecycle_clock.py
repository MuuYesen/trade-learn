"""Fixed BT multi-feed callback/count semantics on declared atomic bar groups."""

import backtrader as official
import pandas as pd

from tradelearn import engine as tl
from tradelearn.backtest.event_runtime import StrategyRuntime


def bars(days):
    return pd.DataFrame(
        {name: [10.0] * len(days) for name in ("open", "high", "low", "close", "volume")},
        index=pd.to_datetime(["2026-01-" + day for day in days]),
    )


def calls(api, stream=False, first_days=None, second_days=None):
    result = []

    class Business(api.Strategy):
        def read(self, phase):
            result.append((phase, len(self), len(self.data0), len(self.data1)))

        def prenext(self):
            self.read("prenext")

        def nextstart(self):
            self.read("nextstart")

        def next(self):
            self.read("next")

    first = bars(first_days or ["01", "03"])
    second = bars(second_days or ["02", "03"])
    if stream:
        runtime = StrategyRuntime(Business, feeds={"A": "bar", "B": "bar"})
        try:
            runtime.start()
            for stamp in first.index.union(second.index):
                group = []
                for symbol, frame in [("A", first), ("B", second)]:
                    if stamp in frame.index:
                        group.append(
                            (
                                symbol,
                                dict(ts=stamp.tz_localize("UTC"), **frame.loc[stamp].to_dict()),
                            )
                        )
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
    return result


def test_stream_engine_warmup_phase_and_strategy_len_match_real_bt_grouped_clock():
    expected = calls(official)
    assert expected == [("prenext", 1, 1, 0), ("nextstart", 2, 1, 1), ("next", 3, 2, 2)]
    assert calls(tl, stream=True) == expected


def test_stream_engine_nextstart_repeats_while_secondary_feed_at_bt_minimum():
    options = {"first_days": ["01", "03", "04"], "second_days": ["02", "04"]}
    expected = calls(official, **options)
    assert [row[0] for row in expected] == ["prenext", "nextstart", "nextstart", "next"]
    assert calls(tl, stream=True, **options) == expected
