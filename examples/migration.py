"""Migration smoke checks for old user-facing concepts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import tradelearn.engine as bt
from tradelearn.ml import CausalSelector
from tradelearn.report import Reporter


@dataclass(frozen=True)
class MigrationCheckpoint:
    identifier: str
    description: str


MIGRATION_CHECKPOINTS = (
    MigrationCheckpoint("import", "Backtrader-style imports use tradelearn.engine."),
    MigrationCheckpoint("line_indexing", "[0] is current bar and [-1] is previous bar."),
    MigrationCheckpoint("cerebro", "Cerebro keeps the event-driven run lifecycle."),
    MigrationCheckpoint("ml", "ML helpers are available from tradelearn.ml."),
    MigrationCheckpoint("report", "Reporter owns customized report generation."),
)


class _BuyOnce(bt.Strategy):
    def next(self) -> None:
        if len(self.data) == 2 and not self.position:
            self.buy(size=1)


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.2, 2.2, 3.2, 4.2],
            "low": [0.8, 1.8, 2.8, 3.8],
            "close": [1.0, 2.0, 3.0, 4.0],
            "volume": [100.0, 100.0, 100.0, 100.0],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
    )


def _line_indexing() -> dict[str, float]:
    data = bt.DataFeed(_bars(), name="migration")
    data._advance(2)
    return {"current": float(data.close[0]), "previous": float(data.close[-1])}


def _cerebro_smoke() -> dict[str, int]:
    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.adddata(_bars(), name="migration")
    cerebro.addstrategy(_BuyOnce)
    [strategy] = cerebro.run()
    return {"fills": int(strategy.stats.summary["total_fills"])}


def run_migration_smoke() -> dict[str, object]:
    """Return a compact map of migration checkpoints."""

    return {
        "checkpoints": [checkpoint.identifier for checkpoint in MIGRATION_CHECKPOINTS],
        "backtrader_import": "tradelearn.engine",
        "line_indexing": _line_indexing(),
        "cerebro": _cerebro_smoke(),
        "ml": {"selector": CausalSelector.__name__},
        "report": {"reporter": Reporter.__name__},
    }


if __name__ == "__main__":
    print(run_migration_smoke())
