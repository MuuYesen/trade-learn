"""Minimal MLStrategy example using Alpha101-style feature preparation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import tradelearn.engine as bt
import tradelearn.factor as tf


@dataclass(frozen=True)
class MLExampleResult:
    selected_features: list[str]
    final_value: float
    stats: object
    factors: pd.DataFrame


class ThresholdModel:
    """Tiny deterministic model for examples and tests."""

    def fit(self, X: list[list[float]], y: list[float]) -> ThresholdModel:
        values = [row[0] for row in X if row]
        self.threshold_ = sum(values) / len(values) if values else 0.0
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        value = X[0][0] if X and X[0] else 0.0
        return [1.0 if value >= getattr(self, "threshold_", 0.0) else -1.0]


def build_alpha101_features(bars: pd.DataFrame, max_features: int = 3) -> pd.DataFrame:
    """Build a compact Alpha101 feature frame aligned to ``bars.index``."""
    stock_data = bars.copy()
    stock_data["date"] = stock_data.index
    stock_data["code"] = "DEMO"
    stock_data["vwap"] = (
        stock_data[["open", "high", "low", "close"]].mean(axis=1)
        if "vwap" not in stock_data
        else stock_data["vwap"]
    )
    names = ["alpha001", "alpha002", "alpha003", "alpha004", "alpha005"][:max_features]
    factors = tf.alpha101(stock_data, names=names)
    feature_frame = factors.set_index("date").drop(columns=["code"])
    feature_frame.columns = [name.replace("_101", "") for name in feature_frame.columns]
    return feature_frame.reindex(bars.index)


def run_example() -> MLExampleResult:
    """Run a deterministic MLStrategy demo and return reportable results."""
    bars = _demo_bars()
    factors = build_alpha101_features(bars, max_features=2)
    selected_features = list(factors.columns)
    data = pd.concat([bars, factors], axis=1)
    data["target"] = data["close"].pct_change().shift(-1).fillna(0.0)

    class AlphaStrategy(bt.MLStrategy):
        model = ThresholdModel()
        features = tuple(selected_features)
        target = "target"

    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.adddata(data, name="alpha")
    cerebro.addstrategy(AlphaStrategy, threshold=0.0, size=1)
    [strategy] = cerebro.run()

    return MLExampleResult(
        selected_features=selected_features,
        final_value=float(strategy.stats.summary["final_value"]),
        stats=strategy.stats,
        factors=factors,
    )


def _demo_bars() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=90, tz="UTC")
    close = pd.Series([20.0 + i * 0.05 + (i % 7) * 0.03 for i in range(len(index))], index=index)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]) * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": [10_000.0 + i * 10 for i in range(len(index))],
        },
        index=index,
    )


if __name__ == "__main__":
    print(run_example())
