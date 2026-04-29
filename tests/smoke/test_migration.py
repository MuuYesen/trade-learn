"""Executable migration checkpoints for the Stage 9 migration guide."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

import tradelearn.engine as bt
from examples.backtrader import MigratedSmaCross
from tradelearn.backtest import LineSeries
from tradelearn.factor import FactorAnalyzer
from tradelearn.ml import CausalSelector, MLStrategy
from tradelearn.report import Reporter


@dataclass(frozen=True)
class MigrationCheckpoint:
    """One documented 1.x to 2.0 migration checkpoint."""

    identifier: str
    legacy: str
    replacement: str


MIGRATION_CHECKPOINTS: tuple[MigrationCheckpoint, ...] = (
    MigrationCheckpoint("query_import", "tradelearn.query.Query", "tradelearn.data"),
    MigrationCheckpoint("backtest_runner", "Backtest(data, Strategy).run()", "bt.Cerebro"),
    MigrationCheckpoint("strategy_init", "Strategy.init", "Strategy.__init__"),
    MigrationCheckpoint("line_current", "self.data.close[-1]", "self.data.close[0]"),
    MigrationCheckpoint("params", "class attributes", "params + self.p"),
    MigrationCheckpoint("indicator_registration", "self.I(func, ...)", "bt.talib / bt.tdx / bt.tv"),
    MigrationCheckpoint("factor_analyzer", "strategy.examine", "tradelearn.factor.FactorAnalyzer"),
    MigrationCheckpoint("reporter", "Evaluate.analysis_report", "tradelearn.report.Reporter"),
    MigrationCheckpoint("ml", "AutoML / CausalGraph", "tradelearn.ml.MLStrategy"),
)

def run_migration_smoke() -> dict[str, Any]:
    """Run compact checks that back the migration guide examples.runners."""
    line = LineSeries([1.0, 2.0, 3.0])
    line._advance(2)

    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.adddata(bt.feeds.PandasData(dataname=_bars(), name="demo"))
    cerebro.addstrategy(MigratedSmaCross, size=2)
    [strategy] = cerebro.run()
    if strategy.stats is None:
        raise RuntimeError("migration smoke strategy did not produce stats")

    return {
        "checkpoints": [checkpoint.identifier for checkpoint in MIGRATION_CHECKPOINTS],
        "backtrader_import": bt.__name__,
        "line_indexing": {"current": line[0], "previous": line[-1]},
        "cerebro": {
            "strategy": strategy.__class__.__name__,
            "fills": len(strategy.stats.fills),
            "final_value": float(strategy.stats.summary["final_value"]),
        },
        "factor": {"analyzer": FactorAnalyzer.__name__},
        "ml": {"strategy": MLStrategy.__name__, "selector": CausalSelector.__name__},
        "report": {"reporter": Reporter.__name__},
    }


def _bars() -> pd.DataFrame:
    close = [
        10.0,
        10.4,
        10.1,
        10.9,
        11.6,
        10.8,
        10.0,
        9.5,
        10.3,
        11.4,
        12.1,
        11.0,
    ]
    return pd.DataFrame(
        {
            "open": [value - 0.1 for value in close],
            "high": [value + 0.5 for value in close],
            "low": [value - 0.5 for value in close],
            "close": close,
            "volume": [1000.0 + index * 10.0 for index in range(len(close))],
        },
        index=pd.date_range("2026-02-01", periods=len(close), freq="D", tz="UTC"),
    )
