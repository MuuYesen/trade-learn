import numpy as np
import pandas as pd


def test_optional_components_use_validation_and_leave_test_unconsumed():
    from examples.research.components import run_components

    dates = pd.date_range("2026-01-01", periods=60, tz="UTC")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["timestamp", "symbol"])
    price = np.array([100 + i + 3 * np.sin(i / 3 + j) for i in range(60) for j in range(2)])
    bars = pd.DataFrame({field: price for field in ["open", "high", "low", "close"]}, index=index)
    bars["volume"] = 100.0
    result = run_components(bars, train_end=dates[30], validation_end=dates[48])
    assert len(result.artifacts["trials"]) == 3
    assert result.artifacts["test_consumed"] is False
    assert result.features.index.get_level_values("timestamp").max() < dates[48]
    assert set(result.artifacts["assessment"]) == {"1", "2"}
    assert result.artifacts["symbolic"]["candidate_capture"] == "complete"
    assert result.weights.sum() == __import__("pytest").approx(0.8)

    changed = bars.copy()
    changed.loc[
        changed.index.get_level_values("timestamp") >= dates[48], ["open", "high", "low", "close"]
    ] *= 3.0
    repeated = run_components(changed, train_end=dates[30], validation_end=dates[48])
    pd.testing.assert_series_equal(result.scores, repeated.scores)
    pd.testing.assert_series_equal(result.weights, repeated.weights)
    assert [t["params"] for t in result.artifacts["trials"]] == [
        t["params"] for t in repeated.artifacts["trials"]
    ]
    assert [t["value"] for t in result.artifacts["trials"]] == [
        t["value"] for t in repeated.artifacts["trials"]
    ]
