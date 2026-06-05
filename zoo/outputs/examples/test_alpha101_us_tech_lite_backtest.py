import pandas as pd
from alpha101_us_tech_lite_backtest import build_composite_scores, build_target_weights


def test_build_composite_scores_returns_timestamp_symbol_series() -> None:
    factors = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "alpha001_101": [1.0, 2.0, 3.0, 1.0],
        }
    )
    selected = pd.DataFrame({"column": ["alpha001_101"], "direction": [1]})

    scores = build_composite_scores(factors, selected)

    assert scores.index.names == ["date", "symbol"]
    assert scores.name == "score"


def test_build_target_weights_returns_timestamp_symbol_series() -> None:
    factors = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "alpha001_101": [1.0, 2.0, 3.0, 1.0],
        }
    )
    selected = pd.DataFrame({"column": ["alpha001_101"], "direction": [1]})

    weights = build_target_weights(factors, selected, rebalance_every=1)

    assert weights.index.names == ["timestamp", "symbol"]
    assert weights.name == "weight"
    assert weights.index.get_level_values("timestamp").tz is None
    assert weights.groupby(level="timestamp").sum().round(10).isin([0.0, 1.0]).all()
