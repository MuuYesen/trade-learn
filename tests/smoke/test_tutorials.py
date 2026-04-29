"""Runnable smoke examples.runners backing the Stage 9 tutorial pages."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pandas as pd

import tradelearn.engine as bt
from examples.backtrader import QuickstartSmaCross
from scripts.examples.ml_strategy import prepare_ml_data as build_alpha101_features
from scripts.examples.ml_strategy import run_example as run_ml_example
from tradelearn.engine.analyzers import MLflowAnalyzer
from tradelearn.core.config import TradelearnConfig
from tradelearn.lab import build_lab_plan


def run_quickstart() -> dict[str, Any]:
    """Run the quickstart strategy and return a compact result summary."""
    bars = _sample_bars()
    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.adddata(bt.feeds.PandasData(dataname=bars, name="demo"))
    cerebro.addstrategy(QuickstartSmaCross)

    [strategy] = cerebro.run()
    if strategy.stats is None:
        raise RuntimeError("quickstart strategy did not produce stats")

    summary = strategy.stats.summary
    return {
        "strategy": QuickstartSmaCross.__name__,
        "bars": len(bars),
        "fills": len(strategy.stats.fills),
        "final_value": float(summary["final_value"]),
        "return_pct": float(strategy.stats.returns.add(1.0).prod() - 1.0),
    }


def _sample_bars() -> pd.DataFrame:
    close = [
        10.0,
        10.3,
        10.1,
        10.8,
        11.4,
        10.7,
        9.9,
        9.4,
        10.2,
        11.1,
        12.0,
        11.2,
        10.4,
        9.8,
        10.5,
        11.5,
        12.6,
        11.7,
        10.8,
        10.0,
        10.9,
        12.0,
        13.1,
        12.2,
        11.1,
        10.3,
        11.3,
        12.5,
        13.8,
        12.6,
    ]
    return pd.DataFrame(
        {
            "open": [value - 0.2 for value in close],
            "high": [value + 0.6 for value in close],
            "low": [value - 0.7 for value in close],
            "close": close,
            "volume": [1000.0 + index * 25.0 for index in range(len(close))],
        },
        index=pd.date_range("2026-01-01", periods=len(close), freq="D", tz="UTC"),
    )


def run_tutorial_smoke() -> dict[str, dict[str, Any]]:
    """Run compact examples.runners for every Stage 9 tutorial topic."""
    return {
        "factor_research": run_factor_research_tutorial(),
        "strategy_backtest": run_strategy_backtest_tutorial(),
        "portfolio": run_portfolio_tutorial(),
        "ml_strategy": run_ml_strategy_tutorial(),
        "mlflow": run_mlflow_tutorial(),
        "jupyterlab": run_jupyterlab_tutorial(),
        "backtrader_migration": run_backtrader_migration_tutorial(),
    }


def run_factor_research_tutorial() -> dict[str, Any]:
    """Compute Alpha101 features for a small deterministic OHLCV sample."""
    factors = build_alpha101_features(_sample_bars(), max_features=2)
    return {
        "rows": len(factors),
        "feature_count": len(factors.columns),
        "features": list(factors.columns),
    }


def run_strategy_backtest_tutorial() -> dict[str, Any]:
    """Run the quickstart backtest tutorial."""
    return run_quickstart()


def run_portfolio_tutorial() -> dict[str, Any]:
    """Run a two-data buy-and-hold example to exercise portfolio artifacts."""
    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.adddata(bt.feeds.PandasData(dataname=_sample_bars(), name="asset_a"))
    cerebro.adddata(bt.feeds.PandasData(dataname=_sample_bars() * 1.03, name="asset_b"))
    cerebro.addstrategy(_TwoAssetBuyAndHold)

    [strategy] = cerebro.run()
    if strategy.stats is None:
        raise RuntimeError("portfolio tutorial did not produce stats")
    return {
        "final_value": float(strategy.stats.summary["final_value"]),
        "position_rows": len(strategy.stats.positions),
        "fills": len(strategy.stats.fills),
    }


def run_ml_strategy_tutorial() -> dict[str, Any]:
    """Run the sklearn GBM + Alpha101 MLStrategy tutorial."""
    result = run_ml_example()
    return {
        "selected_features": result.selected_features,
        "final_value": result.final_value,
        "fills": len(result.stats.fills),
    }


def run_mlflow_tutorial() -> dict[str, Any]:
    """Run a local MLflowAnalyzer smoke without contacting a remote server."""
    fake_mlflow = _FakeMLflow()
    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.adddata(bt.feeds.PandasData(dataname=_sample_bars(), name="demo"))
    cerebro.addstrategy(QuickstartSmaCross)
    cerebro.addanalyzer(
        MLflowAnalyzer,
        name="mlflow",
        experiment="tutorial",
        run_name="quickstart",
        uri="file:///tmp/tradelearn-mlflow-tutorial",
        mlflow_module=fake_mlflow,
    )

    [strategy] = cerebro.run()
    return {
        "status": strategy.analyzer_results["mlflow"]["status"],
        "metric_count": len(fake_mlflow.metrics),
        "artifact_count": len(fake_mlflow.artifacts),
    }


def run_jupyterlab_tutorial() -> dict[str, Any]:
    """Build the JupyterLab dry-run plan used by the lab tutorial."""
    plan = build_lab_plan(
        TradelearnConfig(mlflow_tracking_uri="file:///tmp/tradelearn-mlflow"),
        no_browser=True,
        missing_packages=("jupyter-ai",),
    )
    return {
        "jupyter_command": list(plan.jupyter.args),
        "mcp_command": list(plan.mcp.args),
        "missing_packages": list(plan.missing_packages),
    }


def run_backtrader_migration_tutorial() -> dict[str, Any]:
    """Run a migrated backtrader-style strategy through the compat facade."""
    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.adddata(bt.feeds.PandasData(dataname=_sample_bars(), name="demo"))
    cerebro.addstrategy(QuickstartSmaCross)

    [strategy] = cerebro.run()
    if strategy.stats is None:
        raise RuntimeError("backtrader migration tutorial did not produce stats")
    return {
        "strategy": QuickstartSmaCross.__name__,
        "fills": len(strategy.stats.fills),
        "final_value": float(strategy.stats.summary["final_value"]),
    }


class _TwoAssetBuyAndHold(bt.Strategy):
    def __init__(self) -> None:
        self._entered = False

    def next(self) -> None:
        if self._entered:
            return
        self.buy(data=self.datas[0], size=1)
        self.buy(data=self.datas[1], size=1)
        self._entered = True


class _FakeMLflow:
    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}
        self.artifacts: dict[str, Any] = {}

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, experiment: str) -> None:
        self.experiment = experiment

    @contextmanager
    def start_run(self, **kwargs: Any):
        self.run_kwargs = kwargs
        yield self

    def log_params(self, params: dict[str, Any]) -> None:
        self.params = dict(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update(metrics)

    def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
        self.artifacts[artifact_file] = payload
