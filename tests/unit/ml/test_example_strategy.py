from __future__ import annotations

import pandas as pd
import pytest

from scripts.examples.ml_strategy import build_alpha101_features, prepare_ml_data, run_example


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


def test_prepare_ml_data_uses_current_selector_contract() -> None:
    bars = pd.DataFrame(
        {
            name: [100.0 + i for i in range(80)]
            for name in ("open", "high", "low", "close", "volume")
        },
        index=pd.date_range("2024-01-01", periods=80),
    )
    data, features = prepare_ml_data(bars)
    assert features
    assert data.index.equals(bars.index)
    assert data["target"].iloc[-1] != data["target"].iloc[-1]  # unknown next return
    assert data["target"].iloc[0] == bars["close"].pct_change().iloc[1]


def test_example_holds_out_future_bars_from_training() -> None:
    result = run_example()
    assert result.training_data.index.max() < result.stats.equity.index.min()
    assert result.training_data["target"].notna().all()
    assert len(result.stats.equity) > 0


@pytest.mark.filterwarnings("error:X does not have valid feature names:UserWarning")
def test_example_preserves_feature_names_during_prediction() -> None:
    assert run_example().selected_features


def test_factor_builder_preserves_unknown_warmup_values() -> None:
    from scripts.examples.ml_strategy import _demo_bars

    factors = build_alpha101_features(_demo_bars())
    assert pd.isna(factors["alpha001_101"].iloc[0])
    assert factors["alpha001_101"].notna().any()


def test_missing_features_skip_signals_without_dropping_market_bars() -> None:
    from sklearn.ensemble import GradientBoostingRegressor

    from examples.engine import Alpha101GBMStrategy
    from scripts.examples.ml_strategy import _demo_bars
    from tradelearn.engine import Cerebro

    bars = _demo_bars().iloc[:5].copy()
    bars["feature"] = float("nan")
    training = pd.DataFrame(
        {"feature": [float("nan"), 0.0, 1.0, 2.0], "target": [0.1, 0.1, 0.1, 0.1]}
    )
    cerebro = Cerebro(trade_on_close=True)
    cerebro.adddata(bars)
    cerebro.addstrategy(
        Alpha101GBMStrategy,
        features=("feature",),
        training_data=training,
        model=GradientBoostingRegressor(random_state=7, n_estimators=2),
    )
    [strategy] = cerebro.run()
    assert strategy.stats.equity.index.equals(bars.index)
    assert strategy.stats.fills.empty


def test_holdout_prices_do_not_change_training_factors_labels_or_model() -> None:
    from scripts.examples.ml_strategy import _demo_bars

    bars = _demo_bars()
    baseline = run_example(bars)
    split = int(len(bars) * 0.7)
    changed = bars.copy()
    changed.iloc[split:, changed.columns.get_indexer(["open", "high", "low", "close"])] *= 4
    changed_result = run_example(changed)
    pd.testing.assert_frame_equal(baseline.training_data, changed_result.training_data)
    assert baseline.selected_features == changed_result.selected_features
    assert baseline.stats.equity.index.equals(bars.index[split:])
    assert baseline.training_data.index[-1] == bars.index[split - 2]
    inputs = baseline.training_data[baseline.selected_features].dropna()
    assert (
        baseline.strategy.model_.predict(inputs) == changed_result.strategy.model_.predict(inputs)
    ).all()


def test_missing_price_does_not_create_zero_or_bridged_training_labels() -> None:
    from scripts.examples.ml_strategy import _demo_bars

    bars = _demo_bars().iloc[:40].copy()
    bars.loc[bars.index[20], "close"] = float("nan")
    data, _ = prepare_ml_data(bars)
    assert data["target"].iloc[19:21].isna().all()
    assert pd.isna(data["target"].iloc[-1])


def test_single_asset_undefined_correlations_remain_unknown() -> None:
    from scripts.examples.ml_strategy import _demo_bars

    factors = build_alpha101_features(_demo_bars())
    assert factors[["alpha002_101", "alpha003_101"]].isna().all().all()


def test_alpha_correlations_keep_warmup_missing_and_compute_with_cross_section() -> None:
    import numpy as np

    from tradelearn.factor.alpha import alpha101

    rng = np.random.default_rng(11)
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=30), ["A", "B", "C"]],
        names=["date", "symbol"],
    )
    raw = pd.DataFrame(index=index)
    raw["open"] = rng.uniform(10, 20, len(raw))
    raw["close"] = rng.uniform(10, 20, len(raw))
    raw["high"] = raw[["open", "close"]].max(axis=1)
    raw["low"] = raw[["open", "close"]].min(axis=1)
    raw["vwap"] = (raw["open"] + raw["close"]) / 2
    raw["volume"] = rng.uniform(100, 1000, len(raw))
    factors = alpha101(raw.reset_index(), names=["alpha002", "alpha003"])
    for column, warmup in [("alpha002_101", 7), ("alpha003_101", 9)]:
        dates = factors["date"].drop_duplicates().sort_values()
        assert factors.loc[factors["date"].isin(dates.iloc[:warmup]), column].isna().all()
        assert factors.loc[factors["date"].isin(dates.iloc[warmup:]), column].notna().any()
