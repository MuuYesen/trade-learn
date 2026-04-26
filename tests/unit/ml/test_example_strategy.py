from __future__ import annotations

import pandas as pd

from examples.ml_strategy import build_alpha101_features, run_example


def test_build_alpha101_features_returns_selected_feature_frame() -> None:
    bars = pd.DataFrame(
        {
            "open": [10.0 + index for index in range(80)],
            "high": [11.0 + index for index in range(80)],
            "low": [9.0 + index for index in range(80)],
            "close": [10.5 + index for index in range(80)],
            "volume": [1_000.0 + index for index in range(80)],
        },
        index=pd.date_range("2024-01-01", periods=80, tz="UTC"),
    )

    features = build_alpha101_features(bars, max_features=2)

    assert list(features.columns)
    assert len(features.columns) <= 2
    assert features.index.equals(bars.index)
    assert all(name.startswith("alpha") for name in features.columns)


def test_ml_strategy_example_runs_deterministically() -> None:
    first = run_example()
    second = run_example()

    assert first.selected_features
    assert first.selected_features == second.selected_features
    assert first.final_value == second.final_value
    assert first.stats.summary["final_value"] == first.final_value
    assert not first.stats.equity.empty
    assert isinstance(first.factors, pd.DataFrame)
