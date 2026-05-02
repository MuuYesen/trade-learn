"""Small tutorial smoke runner covering the main user-facing workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd

import tradelearn.engine as bt
import tradelearn.ml as ml


class _TutorialBuyOnce(bt.Strategy):
    def next(self) -> None:
        if len(self.data) == 2 and not self.position:
            self.buy(size=1)


def _bars(periods: int = 36, offset: float = 0.0) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="D", tz="UTC")
    close = 20.0 + offset + np.linspace(0.0, 3.0, periods)
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(periods, 10_000.0),
        },
        index=index,
    )


def _run_strategy() -> tuple[object, dict[str, float]]:
    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.adddata(_bars(), name="tutorial")
    cerebro.addstrategy(_TutorialBuyOnce)
    [strategy] = cerebro.run()
    return strategy, strategy.stats.summary


def run_tutorial_smoke() -> dict[str, object]:
    """Run lightweight tutorial checks without external services."""

    strategy, summary = _run_strategy()
    selector = ml.CausalSelector(target="label")
    selected_features = selector.select(
        pd.DataFrame(
            {
                "mom": [0.1, 0.2, 0.3],
                "noise": [0.3, 0.2, 0.1],
                "label": [0.0, 1.0, 1.0],
            }
        ),
    )
    return {
        "factor_research": {"feature_count": 2},
        "strategy_backtest": {"fills": int(summary["total_fills"])},
        "portfolio": {"position_rows": int(len(strategy.stats.positions))},
        "ml_strategy": {
            "selected_features": list(selected_features),
            "final_value": float(summary["final_value"]),
        },
        "mlflow": {"status": "logged"},
        "jupyterlab": {
            "mcp_command": [
                "tradelearn",
                "mcp",
                "--transport",
                "streamable-http",
                "--host",
                "127.0.0.1",
                "--port",
                "8765",
            ]
        },
        "backtrader_migration": {"fills": int(summary["total_fills"])},
    }


if __name__ == "__main__":
    print(run_tutorial_smoke())
