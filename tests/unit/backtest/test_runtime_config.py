from __future__ import annotations

import pandas as pd

import tradelearn.engine as bt
import tradelearn.lite as tl
from tradelearn.backtest.runtime_config import BacktestRuntimeConfig


def _data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.0, 11.0, 12.0],
            "volume": [100.0, 100.0, 100.0],
        },
        index=pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC"),
    )


def test_runtime_config_is_internal_and_shared_by_lite() -> None:
    class Strategy(tl.Strategy):
        def next(self) -> None:
            pass

    backtest = tl.Backtest(_data(), Strategy, cash=1234.0, commission=0.001, trade_on_close=True)

    config = BacktestRuntimeConfig.from_owner(backtest)

    assert config.cash == 1234.0
    assert config.commission == 0.001
    assert config.trade_on_close is True
    assert config.match_mode == "exact"
    assert not hasattr(tl, "BacktestRuntimeConfig")


def test_runtime_config_is_internal_and_shared_by_engine() -> None:
    cerebro = bt.Cerebro(trade_on_close=True, match_mode="smart", exactbars=True, stdstats=False)
    cerebro.broker.setcash(4321.0)
    cerebro.broker.setcommission(commission=0.002)

    config = BacktestRuntimeConfig.from_owner(cerebro)

    assert config.cash == 4321.0
    assert config.commission == 0.002
    assert config.trade_on_close is True
    assert config.match_mode == "smart"
    assert config.exactbars is True
    assert config.stdstats is False
    assert not hasattr(bt, "BacktestRuntimeConfig")
