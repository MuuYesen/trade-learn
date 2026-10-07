"""Deterministic Alpha101/GBM example with a chronological evaluation holdout.

The default bars are synthetic demonstration data, not historical GOOG prices.
Pass an OHLCV frame to run_example to use real market data instead.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from examples.engine import Alpha101GBMStrategy
from tradelearn.engine import Cerebro
from tradelearn.factor.alpha import alpha101
from tradelearn.ml import CausalSelector


def build_alpha101_features(bars: pd.DataFrame, max_features: int = 3) -> pd.DataFrame:
    """Compute bounded Alpha101 features; preserve the index and unknown values."""
    if not 1 <= max_features <= 3:
        raise ValueError("max_features must be between 1 and 3")
    factor_input = bars.reset_index(names="date").copy()
    dates = pd.DatetimeIndex(factor_input["date"])
    factor_input["date"] = dates.tz_localize(None)
    factor_input["symbol"] = "DEMO"
    factor_input["vwap"] = factor_input[["open", "high", "low", "close"]].mean(axis=1)
    alpha_frame = alpha101(factor_input, names=["alpha001", "alpha002", "alpha003"][:max_features])
    factors = alpha_frame.drop(columns=["symbol"]).set_index("date")
    factors = factors.reindex(dates.tz_localize(None))
    factors.index = bars.index
    return factors.replace([np.inf, -np.inf], np.nan)


def prepare_ml_data(bars: pd.DataFrame):
    """Prepare candidate factors and next-bar labels; leave the last label unknown."""
    factors = build_alpha101_features(bars)
    data = bars.join(factors)
    data["target"] = bars["close"].pct_change(fill_method=None).shift(-1)
    return data, list(factors.columns)


@dataclass
class MLExampleResult:
    selected_features: list[str]
    final_value: float
    stats: Any
    factors: pd.DataFrame
    training_data: pd.DataFrame
    strategy: Any


def _demo_bars() -> pd.DataFrame:
    """Seeded synthetic OHLCV for an offline, reproducible behavior example."""
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0.08, 1.0, 200))
    opening = close + rng.normal(0, 0.2, len(close))
    return pd.DataFrame(
        {
            "open": opening,
            "high": np.maximum(opening, close) + 0.5,
            "low": np.minimum(opening, close) - 0.5,
            "close": close,
            "volume": rng.integers(1000, 10000, len(close)).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=len(close), tz="UTC"),
    )


def run_example(bars: pd.DataFrame | None = None) -> MLExampleResult:
    bars = _demo_bars() if bars is None else bars.copy()
    if len(bars) < 30 or not bars.index.is_monotonic_increasing or not bars.index.is_unique:
        raise ValueError("bars must contain at least 30 unique chronological observations")
    data, candidates = prepare_ml_data(bars)
    split = int(len(data) * 0.7)
    # Purge the boundary label: its next-bar price belongs to the evaluation set.
    training = data.iloc[: split - 1].dropna(subset=["target"])
    usable_candidates = [name for name in candidates if training[name].notna().any()]
    if not usable_candidates:
        raise ValueError("training window has no observed Alpha101 features")
    selector = CausalSelector(target="target", features=usable_candidates, max_features=3)
    features = selector.select(training)
    cerebro = Cerebro()
    cerebro.adddata(data.iloc[split:], name="DEMO")
    cerebro.addstrategy(
        Alpha101GBMStrategy,
        model=GradientBoostingRegressor(random_state=7, n_estimators=50, max_depth=3),
        features=tuple(features),
        threshold=0.001,
        size=10,
        training_data=training,
    )
    [strategy] = cerebro.run()
    return MLExampleResult(
        features,
        float(strategy.stats.summary["final_value"]),
        strategy.stats,
        data.loc[:, candidates],
        training,
        strategy,
    )


def main():
    result = run_example()
    print("Synthetic demonstration data; chronological holdout evaluation.")
    print("\nML Backtest Summary:")
    print(f"Final Value: {result.final_value:.2f}")


if __name__ == "__main__":
    main()
