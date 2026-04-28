"""Runnable example for ML Strategy using Alpha101 and Causal Selection."""

import pandas as pd

from examples.backtrader import Alpha101GBMStrategy
from tradelearn.backtest import Cerebro
from tradelearn.factor.alpha import alpha101
from tradelearn.ml import CausalSelector


def prepare_ml_data(bars: pd.DataFrame):
    """Generate factors and select causal features."""
    # 1. Create Alpha101 input format
    factor_input = bars.reset_index(names="date").copy()
    factor_input["date"] = factor_input["date"].dt.tz_convert(None)
    factor_input["code"] = "DEMO"
    factor_input["vwap"] = (
        factor_input["open"] + factor_input["high"] + factor_input["low"] + factor_input["close"]
    ) / 4.0

    # 2. Calculate factors
    alpha_frame = alpha101(factor_input, names=["alpha001", "alpha002", "alpha003"])
    factors = alpha_frame.drop(columns=["code"]).set_index("date")
    factors.index = pd.DatetimeIndex(factors.index).tz_localize("UTC")

    # 3. Align with bars
    factors = factors.reindex(bars.index).ffill().fillna(0.0)
    target = bars["close"].pct_change().shift(-1).fillna(0.0)

    # 4. Causal Selection
    selector = CausalSelector(max_features=3)
    selected_factors = selector.fit_transform(factors, target)

    # 5. Join all
    data = bars.join(selected_factors)
    data["target"] = target
    return data, list(selected_factors.columns)


def run_example():
    # 1. Load Data
    try:
        bars = pd.read_parquet("tests/data/GOOG.parquet")
    except FileNotFoundError:
        return None

    # 2. Prepare ML Data (Factors + Selection)
    data, features = prepare_ml_data(bars)

    # 3. Setup Strategy
    Alpha101GBMStrategy.features = tuple(features)

    # 4. Run Backtest
    cerebro = Cerebro()
    cerebro.adddata(data, name="GOOG")
    cerebro.addstrategy(Alpha101GBMStrategy, threshold=0.001, size=10, training_data=data)

    strategies = cerebro.run()
    return strategies[0]


def main():
    result = run_example()
    if result is None:
        print("Demo data not found. Run 'python scripts/generate_demo_data.py' first.")
        return

    print("\nML Backtest Summary:")
    print(f"Final Value: {result.broker.getvalue():.2f}")
    print(f"Total Return: {(result.broker.getvalue() / 100000.0 - 1) * 100:.2f}%")


if __name__ == "__main__":
    main()
