"""End-to-end MLStrategy example with Alpha101 features and CausalSelector."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from tradelearn.backtest import Cerebro, Stats
from tradelearn.factor.alpha import alpha101
from tradelearn.ml import CausalSelector, MLStrategy


@dataclass
class MLExampleResult:
    """Result bundle returned by the runnable example."""

    selected_features: list[str]
    factors: pd.DataFrame
    stats: Stats
    final_value: float


class Alpha101GBMStrategy(MLStrategy):
    """Gradient Boosting strategy over Alpha101 feature columns."""

    model = GradientBoostingRegressor(random_state=7, n_estimators=12, max_depth=2)
    target = "target"


def build_alpha101_features(bars: pd.DataFrame, max_features: int = 3) -> pd.DataFrame:
    """Build Alpha101 factor columns and select causal candidates."""
    factor_input = _alpha101_input(bars)
    alpha_frame = alpha101(factor_input, names=["alpha001", "alpha002", "alpha003"])
    factors = (
        alpha_frame.drop(columns=["code"])
        .rename(columns={"date": "timestamp"})
        .set_index("timestamp")
        .sort_index()
    )
    factors.index = pd.DatetimeIndex(factors.index).tz_localize("UTC")
    factors = factors.reindex(bars.index).ffill().fillna(0.0)
    target = bars["close"].pct_change().shift(-1).fillna(0.0)
    selector = CausalSelector(max_features=max_features)
    return selector.fit_transform(factors, target)


def run_example() -> MLExampleResult:
    """Run the complete sklearn GBM + Alpha101 + causal selection example."""
    bars = _sample_bars()
    factors = build_alpha101_features(bars)
    data = bars.join(factors)
    data["target"] = data["close"].pct_change().shift(-1).fillna(0.0)

    Alpha101GBMStrategy.features = tuple(factors.columns)
    cerebro = Cerebro(trade_on_close=True)
    cerebro.adddata(data, name="alpha101-gbm")
    cerebro.addstrategy(Alpha101GBMStrategy, threshold=0.001, size=1, training_data=data)
    [strategy] = cerebro.run()
    if strategy.stats is None:
        raise RuntimeError("example strategy did not produce stats")
    return MLExampleResult(
        selected_features=list(factors.columns),
        factors=factors,
        stats=strategy.stats,
        final_value=float(strategy.stats.summary["final_value"]),
    )


def _sample_bars() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=160, tz="UTC")
    rows = []
    for day, _timestamp in enumerate(index, start=1):
        trend = 20.0 + day * 0.12
        cycle = ((day % 7) - 3) * 0.08
        close = trend + cycle
        rows.append(
            {
                "open": close - 0.10,
                "high": close + 0.35,
                "low": close - 0.35,
                "close": close,
                "volume": 10_000.0 + day * 25.0,
            }
        )
    return pd.DataFrame(rows, index=index)


def _alpha101_input(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.reset_index(names="date").copy()
    frame["date"] = frame["date"].dt.tz_convert(None)
    frame["code"] = "DEMO"
    frame["vwap"] = (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0
    return frame[["date", "code", "open", "high", "low", "close", "volume", "vwap"]]


if __name__ == "__main__":
    result = run_example()
    print(
        {
            "selected_features": result.selected_features,
            "final_value": result.final_value,
            "total_fills": result.stats.summary["total_fills"],
        }
    )
