from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import tradelearn.engine as bt
from tradelearn import metrics
from tradelearn.lite import Backtest, Strategy
from tradelearn.ml import MLStrategy


def _bars(mult: float = 1.0) -> pd.DataFrame:
    close = [10.0, 10.5, 11.0, 10.8, 11.6, 12.0, 11.7, 12.4]
    return pd.DataFrame(
        {
            "open": [(value - 0.1) * mult for value in close],
            "high": [(value + 0.4) * mult for value in close],
            "low": [(value - 0.5) * mult for value in close],
            "close": [value * mult for value in close],
            "volume": [100.0 + i for i in range(len(close))],
            "feature": [i / 10 for i in range(len(close))],
            "target": [0, 1, 1, 0, 1, 1, 0, 1],
        },
        index=pd.date_range("2026-01-01", periods=len(close), freq="D", tz="UTC"),
    )


class EngineBuyAndHold(bt.Strategy):
    def __init__(self) -> None:
        self.entered = False

    def next(self) -> None:
        if not self.entered:
            self.buy(data=self.datas[0], size=1)
            if len(self.datas) > 1:
                self.buy(data=self.datas[1], size=1)
            self.entered = True


class LiteBuyAndHold(Strategy):
    def init(self) -> None:
        self.entered = False

    def next(self) -> None:
        if not self.entered:
            self.buy(size=1)
            self.entered = True


def test_engine_plot_and_html_use_reporter_after_run(tmp_path: Path) -> None:
    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.adddata(_bars(), name="asset_a")
    cerebro.adddata(_bars(1.03), name="asset_b")
    cerebro.addstrategy(EngineBuyAndHold)

    [strategy] = cerebro.run()
    charts = cerebro.plot()
    path = tmp_path / "engine-report.html"
    result = cerebro.report(path)

    assert strategy.stats is not None
    assert strategy.stats.summary["total_fills"] == 2
    assert not strategy.stats.positions.empty
    assert len(charts) == 1
    assert result == path
    assert "Summary Stats" in path.read_text()
    assert "Price / Trades" in path.read_text()
    assert hasattr(cerebro, "report")
    assert not hasattr(cerebro, "html")


def test_lite_plot_and_html_use_shared_reporter_after_run(tmp_path: Path) -> None:
    backtest = Backtest(_bars(), LiteBuyAndHold, cash=10_000, trade_on_close=True)
    stats = backtest.run()

    charts = backtest.plot()
    path = tmp_path / "lite-report.html"
    result = backtest.report(path)

    assert stats["# Trades"] >= 0
    assert len(charts) == 1
    assert result == path
    assert "Summary Stats" in path.read_text()
    assert "Price / Trades" in path.read_text()
    assert hasattr(backtest, "report")
    assert not hasattr(backtest, "html")


def test_oco_multitimeframe_multiasset_ml_and_metrics_surfaces() -> None:
    class BracketAndResample(bt.Strategy):
        def __init__(self) -> None:
            self.orders: list[Any] = []

        def next(self) -> None:
            if not self.orders:
                self.orders = self.buy_bracket(
                    data=self.datas[0],
                    size=1,
                    price=10.0,
                    stopprice=9.0,
                    limitprice=12.0,
                )

    cerebro = bt.Cerebro(trade_on_close=True)
    daily = cerebro.adddata(_bars(), name="daily")
    cerebro.resampledata(daily, timeframe=bt.TimeFrame.Weeks, compression=1, name="weekly")
    cerebro.adddata(_bars(1.01), name="asset_b")
    cerebro.addstrategy(BracketAndResample)
    [strategy] = cerebro.run()

    main, stop, limit = strategy.orders
    assert len(strategy.datas) == 3
    assert stop.parent is main
    assert limit.parent is main
    assert limit.oco is stop
    assert strategy.getdatabyname("daily") is strategy.datas[0]

    class DemoModel:
        def fit(self, X, y):
            self.fit_shape = (len(X), len(X[0]))
            return self

        def predict(self, X):
            return [1.0]

    class DemoML(MLStrategy):
        model = DemoModel()
        features = ("feature",)
        target = "target"

    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.adddata(_bars(), name="ml")
    cerebro.addstrategy(DemoML, threshold=0.5, size=1)
    [ml_strategy] = cerebro.run()

    assert ml_strategy.stats is not None
    assert ml_strategy.model_.fit_shape == (8, 1)
    assert metrics.sharpe(pd.Series([0.01, -0.01, 0.02]), periods=252) == metrics.sharpe(
        pd.Series([0.01, -0.01, 0.02]), periods=252
    )
