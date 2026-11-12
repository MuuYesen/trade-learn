"""Feed periods come from fixed NA BarType declarations, never observed cadence guesses."""

import pytest

pytest.importorskip("nautilus_trader")
from nautilus_trader.model import BarType, InstrumentId

from tradelearn import engine as bt
from tradelearn_runtime.execution import _feed_period_metadata
from tradelearn_runtime.store import _Subscription


@pytest.mark.parametrize(
    "spec,timeframe,compression",
    [
        ("2-MILLISECOND", bt.TimeFrame.MicroSeconds, 2000),
        ("3-SECOND", bt.TimeFrame.Seconds, 3),
        ("5-MINUTE", bt.TimeFrame.Minutes, 5),
        ("2-HOUR", bt.TimeFrame.Minutes, 120),
        ("1-DAY", bt.TimeFrame.Days, 1),
        ("1-WEEK", bt.TimeFrame.Weeks, 1),
        ("1-MONTH", bt.TimeFrame.Months, 1),
        ("1-YEAR", bt.TimeFrame.Years, 1),
    ],
)
def test_native_bar_declaration_supplies_same_subscription_and_runtime_period(
    spec, timeframe, compression
):
    bar = BarType.from_str("EURUSD.SIM-" + spec + "-MID-INTERNAL")
    subscription = _Subscription(object(), bar, "A")
    assert (subscription._timeframe, subscription._compression) == (timeframe, compression)
    assert subscription.p.timeframe == timeframe and subscription.p.compression == compression
    assert _feed_period_metadata({"A": InstrumentId.from_str("EURUSD.SIM")}, [bar]) == {
        "A": dict(timeframe=timeframe, compression=compression)
    }


def test_non_time_bars_do_not_invent_a_daily_period():
    bar = BarType.from_str("EURUSD.SIM-10-VOLUME-MID-INTERNAL")
    subscription = _Subscription(object(), bar, "A")
    assert subscription._timeframe is None and subscription._compression is None


def test_runtime_exposes_declared_period_before_business_constructor():
    from tradelearn.backtest.event_runtime import StrategyRuntime

    calls = []

    class Business(bt.Strategy):
        def __init__(self):
            calls.append(
                (
                    self.data._timeframe,
                    self.data._compression,
                    self.data.p.timeframe,
                    self.data.p.compression,
                )
            )

    runtime = StrategyRuntime(
        Business,
        feeds={"A": "bar"},
        feed_metadata={"A": dict(timeframe=bt.TimeFrame.Minutes, compression=5)},
    )
    assert calls == [(bt.TimeFrame.Minutes, 5, bt.TimeFrame.Minutes, 5)]
    runtime.cleanup()


def test_ambiguous_native_periods_remain_unknown_instead_of_last_writer_winning():
    bars = [
        BarType.from_str("EURUSD.SIM-" + spec + "-MID-INTERNAL")
        for spec in ["1-SECOND", "1-MINUTE"]
    ]
    assert _feed_period_metadata({"A": InstrumentId.from_str("EURUSD.SIM")}, bars) == {}
