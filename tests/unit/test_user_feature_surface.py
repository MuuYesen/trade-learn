from __future__ import annotations

from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

import tradelearn.engine as bt
from tradelearn import metrics
from tradelearn.lite import Backtest, Strategy
from tradelearn.research import ModelScorer


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
    assert "Portfolio Replay" in path.read_text()
    assert hasattr(cerebro, "report")
    assert not hasattr(cerebro, "html")


def test_engine_report_dispatches_to_excel(tmp_path: Path) -> None:
    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.adddata(_bars(), name="asset_a")
    cerebro.addstrategy(EngineBuyAndHold)
    cerebro.run()

    path = tmp_path / "engine-report.xlsx"
    result = cerebro.report(path)

    assert result == path
    assert path.exists()
    with ZipFile(path) as workbook:
        assert "xl/workbook.xml" in workbook.namelist()


def test_lite_plot_and_html_use_shared_reporter_after_run(tmp_path: Path) -> None:
    backtest = Backtest(_bars(), LiteBuyAndHold, cash=10_000, trade_on_close=True)
    stats = backtest.run()

    charts = backtest.plot()
    path = tmp_path / "lite-report.html"
    result = backtest.report(path)

    assert stats["total_trades"] >= 0
    assert len(charts) == 1
    assert result == path
    assert "Summary Stats" in path.read_text()
    assert "Price / Trades" in path.read_text()
    assert hasattr(backtest, "report")
    assert not hasattr(backtest, "html")


def test_lite_report_dispatches_to_excel(tmp_path: Path) -> None:
    backtest = Backtest(_bars(), LiteBuyAndHold, cash=10_000, trade_on_close=True)
    backtest.run()

    path = tmp_path / "lite-report.xlsx"
    result = backtest.report(path)

    assert result == path
    assert path.exists()
    with ZipFile(path) as workbook:
        assert "xl/workbook.xml" in workbook.namelist()


def test_oco_multitimeframe_multiasset_model_scorer_and_metrics_surfaces() -> None:
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
        def predict(self, X):
            return [1.0] * len(X)

    scores = ModelScorer(DemoModel(), features=("feature",)).predict(_bars())

    assert scores.tolist() == [1.0] * len(_bars())
    assert metrics.sharpe(pd.Series([0.01, -0.01, 0.02]), periods=252) == metrics.sharpe(
        pd.Series([0.01, -0.01, 0.02]), periods=252
    )


def test_user_facades_hide_internal_helper_functions() -> None:
    import tradelearn.backtest as backtest
    import tradelearn.data as data
    import tradelearn.factor as factor
    import tradelearn.factor.alpha as alpha

    data_hidden = {
        "bars_fingerprint",
        "infer_tdx_market",
        "normalize_bars",
        "resolve_tdx_symbol",
        "resample_frame",
    }
    backtest_hidden = {
        "_notify_order",
        "run_backtest",
        "BatchIndicatorCache",
        "DelayedLine",
        "IndicatorCache",
        "IndicatorLine",
        "LineSeries",
        "Lines",
        "RollingBarBuffer",
        "RollingIndicatorCache",
        "SharedBarBuffer",
    }
    factor_hidden = {
        "ALPHA101_SKIPPED",
        "ALPHA101_SUPPORTED",
        "ALPHA191_SKIPPED",
        "ALPHA191_SUPPORTED",
        "AlphaFormulaBlocker",
        "AlphaFormulaFamilyMetadata",
        "alpha_formula_blockers",
        "alpha_formula_metadata",
        "validate_alpha_formula_metadata",
        "validated_alpha_formula_metadata",
    }

    assert all(not hasattr(data, name) for name in data_hidden)
    assert data_hidden.isdisjoint(data.__all__)
    assert all(not hasattr(backtest, name) for name in backtest_hidden)
    assert backtest_hidden.isdisjoint(backtest.__all__)
    assert all(not hasattr(factor, name) for name in factor_hidden)
    assert factor_hidden.isdisjoint(factor.__all__)
    assert all(not hasattr(alpha, name) for name in factor_hidden)
    assert factor_hidden.isdisjoint(alpha.__all__)
