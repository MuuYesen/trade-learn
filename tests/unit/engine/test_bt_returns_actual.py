"""Fixed BT Returns profile on observed broker value, separate from common TWR metrics."""

import backtrader as official
import pandas as pd
import pytest

from tradelearn import engine as tl


def returns_calls(api, options, analyzer="Returns", sample_from_feed=False):
    state = {"value": 100.0}
    frame = pd.DataFrame(
        {key: [100.0, 110.0, 90.0, 120.0] for key in ("open", "high", "low", "close", "volume")},
        index=pd.to_datetime(
            ["2026-01-30 10:00", "2026-01-30 10:01", "2026-02-02 10:00", "2026-02-02 10:03"]
        ),
    )

    class Business(api.Strategy):
        def __init__(self):
            self.broker.getvalue = (
                (lambda: float(self.data.close[0]) if len(self.data) else 100.0)
                if sample_from_feed
                else lambda: state["value"]
            )

        def next(self):
            state["value"] = float(self.data.close[0])

    engine = api.Cerebro(stdstats=False)
    engine.broker.setcash(100.0)
    engine.adddata(api.feeds.PandasData(dataname=frame))
    engine.addstrategy(Business)
    kwargs = dict(options)
    if "timeframe" in kwargs:
        kwargs["timeframe"] = getattr(api.TimeFrame, kwargs["timeframe"])
    engine.addanalyzer(getattr(api.analyzers, analyzer), _name="returns", **kwargs)
    return dict(engine.run()[0].analyzers.returns.get_analysis())


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"timeframe": "Days", "tann": 365},
        {"timeframe": "Weeks"},
        {"timeframe": "Months"},
        {"timeframe": "Years"},
        {"timeframe": "Minutes", "compression": 2, "tann": 100},
        {"timeframe": "NoTimeFrame"},
        {"fund": False},
    ],
)
def test_observed_value_returns_fields_defaults_and_time_buckets_match_actual_bt(options):
    expected = returns_calls(official, options)
    assert returns_calls(tl, options) == pytest.approx(expected)


def test_subclass_next_without_super_keeps_returns_period_accounting():
    def result(api):
        state = {"value": 100.0}
        calls = []

        class Business(api.Strategy):
            def __init__(self):
                self.broker.getvalue = lambda: state["value"]

            def next(self):
                state["value"] += 10

        class Custom(api.analyzers.Returns):
            def next(self):
                calls.append(len(self.strategy))

        e = api.Cerebro(stdstats=False)
        e.broker.setcash(100)
        frame = pd.DataFrame(
            {key: [10.0] * 3 for key in ("open", "high", "low", "close", "volume")},
            index=pd.date_range("2026-01-01", periods=3),
        )
        e.adddata(api.feeds.PandasData(dataname=frame))
        e.addstrategy(Business)
        e.addanalyzer(Custom, _name="custom")
        analysis = dict(e.run()[0].analyzers.custom.get_analysis())
        return calls, analysis

    calls, analysis = result(tl)
    expected_calls, expected = result(official)
    assert calls == expected_calls
    assert analysis == pytest.approx(expected)


def test_drawdown_samples_before_user_next_changes_broker_value_like_bt():
    expected = returns_calls(official, {}, analyzer="DrawDown")
    actual = returns_calls(tl, {}, analyzer="DrawDown")
    assert actual["max"] == pytest.approx(dict(expected["max"]))
    assert {k: v for k, v in actual.items() if k != "max"} == pytest.approx(
        {k: v for k, v in expected.items() if k != "max"}
    )
