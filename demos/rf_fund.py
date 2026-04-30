"""Multi-asset fund rotation demo using the Engine API."""

from __future__ import annotations

import numpy as np
import pandas as pd

import tradelearn.engine as bt


class RandomForestRotation(bt.Strategy):
    """Deterministic stand-in for a trained fund rotation workflow."""

    params = (("lookback", 6),)

    def __init__(self) -> None:
        self.probabilities: dict[str, float] = {}
        self.selected_symbol: str | None = None
        self.train_rows = 0

    def next(self) -> None:
        if len(self.data) < self.p.lookback + 1:
            return
        scores: dict[str, float] = {}
        for data in self.datas:
            symbol = str(getattr(data, "_name", None))
            scores[symbol] = float(data.close[0] / data.close[-self.p.lookback] - 1.0)
        total = sum(abs(score) for score in scores.values()) or 1.0
        self.probabilities = {symbol: abs(score) / total for symbol, score in scores.items()}
        self.selected_symbol = max(scores, key=scores.get)
        self.train_rows = len(self.data) - self.p.lookback
        for data in self.datas:
            target = 0.95 if getattr(data, "_name", None) == self.selected_symbol else 0.0
            self.order_target_percent(data=data, target=target)


def _bars(offset: float, trend: float) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=48, freq="D", tz="UTC")
    close = offset + trend * np.arange(len(index)) + np.sin(np.linspace(0.0, 5.0, len(index)))
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(len(index), 5_000.0),
        },
        index=index,
    )


def run_demo() -> dict[str, object]:
    """Run the rotation demo and return a compact summary."""

    symbols = ["asset_a", "asset_b", "asset_c"]
    frames = {
        "asset_a": _bars(20.0, 0.05),
        "asset_b": _bars(18.0, 0.08),
        "asset_c": _bars(22.0, 0.03),
    }
    cerebro = bt.Cerebro(trade_on_close=True)
    for symbol in symbols:
        cerebro.adddata(frames[symbol], name=symbol)
    cerebro.addstrategy(RandomForestRotation)
    [strategy] = cerebro.run()
    return {
        "strategy": strategy.__class__.__name__,
        "symbols": symbols,
        "selected_symbol": strategy.selected_symbol,
        "probabilities": dict(strategy.probabilities),
        "fills": int(strategy.stats.summary["total_fills"]),
        "final_value": float(strategy.stats.summary["final_value"]),
        "train_rows": int(strategy.train_rows),
    }


if __name__ == "__main__":
    print(run_demo())
