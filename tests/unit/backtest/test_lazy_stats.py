from __future__ import annotations

import pandas as pd

from tradelearn.engine import Cerebro, Strategy


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [11.0, 12.0, 13.0, 14.0],
            "low": [9.0, 10.0, 11.0, 12.0],
            "close": [10.0, 11.0, 12.0, 13.0],
            "volume": [100.0, 110.0, 120.0, 130.0],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC"),
    )


def test_engine_lazy_stats_defer_pandas_artifacts_until_access() -> None:
    class BuyOnce(Strategy):
        def next(self) -> None:
            if len(self.data) == 2:
                self.buy(size=1)

    cerebro = Cerebro(stats_mode="lazy")
    cerebro.setcash(1000.0)
    cerebro.adddata(_bars())
    cerebro.addstrategy(BuyOnce)

    [strategy] = cerebro.run()
    stats = strategy.stats

    assert stats.summary["bars"] == 4.0
    assert not stats.is_materialized("equity")
    assert not stats.is_materialized("fills")
    assert not stats.is_materialized("trades")

    assert not stats.equity.empty
    assert stats.is_materialized("equity")
    assert not stats.is_materialized("fills")

    assert not stats.fills.empty
    assert stats.is_materialized("fills")
    assert not stats.is_materialized("trades")

    _ = stats.trades
    assert stats.is_materialized("trades")
