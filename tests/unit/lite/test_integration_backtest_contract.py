"""Regression cases for final Lite integration gaps."""

import concurrent.futures

import pandas as pd
import pytest

from tradelearn.lite import Backtest, Strategy


def bars():
    return pd.DataFrame(
        {
            "open": [10.0, 12.0, 14.0, 16.0],
            "high": [11.0, 13.0, 15.0, 17.0],
            "low": [9.0, 11.0, 13.0, 15.0],
            "close": [10.0, 12.0, 14.0, 16.0],
            "volume": [100.0] * 4,
        },
        index=pd.date_range("2026-01-01", periods=4, tz="UTC"),
    )


class BuyOnce(Strategy):
    amount = 1

    def next(self):
        if len(self) == 1:
            self.buy(size=self.amount)


@pytest.mark.parametrize(
    "options",
    [
        {"margin": 0.5},
        {"hedging": True},
        {"exclusive_orders": True},
        {"holding": {"Asset": 1}},
        {"trade_start_date": "2026-01-02"},
        {"lot_size": 10},
        {"fail_fast": False},
    ],
)
def test_unimplemented_economic_modes_reject_before_run(options):
    with pytest.raises(NotImplementedError, match=next(iter(options))):
        Backtest(bars(), BuyOnce, **options)


def test_grid_keeps_execution_configuration_and_runs_without_serializing_runtime(monkeypatch):
    def no_pool(*args, **kwargs):
        pytest.fail("native runtime must stay in process")

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", no_pool)
    options = dict(
        cash=1000, commission=0.01, trade_on_close=True, match_mode="exact", stats_mode="lazy"
    )
    expected = Backtest(bars(), BuyOnce, **options).run(amount=2)
    actual = Backtest(bars(), BuyOnce, **options).optimize(amount=[1, 2])
    pd.testing.assert_frame_equal(actual.fills, expected.fills)
    pd.testing.assert_series_equal(actual.equity, expected.equity)
    assert actual.strategy.amount == 2
    assert actual.config == expected.config


def test_grid_rejects_shared_mutable_storage():
    with pytest.raises(NotImplementedError, match="storage"):
        Backtest(bars(), BuyOnce, storage={"counter": 1}).optimize(amount=[1, 2])
