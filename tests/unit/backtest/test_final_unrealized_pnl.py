import pandas as pd
import pytest

from tests.unit.backtest._engine_config import configured_engine, feed
from tradelearn.engine import Strategy


def _bars(prices: list[float]) -> pd.DataFrame:
    bars = pd.DataFrame(
        {field: prices for field in ("open", "high", "low", "close")},
        index=pd.date_range("2026-01-01", periods=len(prices), tz="UTC"),
    )
    bars["volume"] = 1000.0
    return bars


@pytest.mark.parametrize("stats_mode", ["full", "lazy"])
@pytest.mark.parametrize(
    ("scenario", "expected_pnl", "fill_prices"),
    [
        ("long", 30.0, [100.0]),
        ("long-and-short", 12.0, [100.0, 200.0]),
        ("closed", 0.0, [100.0, 112.0]),
        ("no-trades", 0.0, []),
        ("stopped-early", 12.0, [100.0]),
        ("missing-last-close", 12.0, [100.0]),
        ("contract-multiplier", 90.0, [100.0]),
    ],
)
def test_summary_uses_the_last_observed_close_for_each_open_position(
    stats_mode, scenario, expected_pnl, fill_prices
):
    class Positions(Strategy):
        def next(self):
            if len(self) == 1 and scenario != "no-trades":
                self.buy(data=self.datas[0], size=1)
                if scenario == "long-and-short":
                    self.sell(data=self.datas[1], size=2)
            if len(self) == 2 and scenario == "closed":
                self.close(data=self.datas[0])
            if len(self) == 3 and scenario == "stopped-early":
                self.cerebro.runstop()

    engine = configured_engine(stats_mode=stats_mode)
    engine.broker.setcash(10_000)
    bars = _bars([90.0, 100.0, 112.0, 130.0])
    if scenario == "missing-last-close":
        bars.loc[bars.index[-1], "close"] = float("nan")
    if scenario == "contract-multiplier":
        engine.broker._configure_commission(mult=3)
    engine.adddata(feed(bars), name="A")
    if scenario == "long-and-short":
        # B's final bar is one day earlier than A's.
        engine.adddata(feed(_bars([180.0, 200.0, 209.0])), name="B")
    engine.addstrategy(Positions)

    [strategy] = engine.run()
    stats = strategy.stats
    assert stats.positions["mark_price"].tolist() == fill_prices
    assert stats.summary["final_unrealized_pnl"] == pytest.approx(expected_pnl)
