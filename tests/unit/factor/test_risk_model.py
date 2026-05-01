"""Tests for factor risk and attribution helpers."""

import math

import pandas as pd

from tradelearn.factor import FactorRiskModel, PerformanceAttribution


def test_factor_risk_model_computes_exposure_risk_and_contributions() -> None:
    exposures = pd.DataFrame(
        {
            "value": {"AAA": 1.0, "BBB": -0.5},
            "momentum": {"AAA": 0.2, "BBB": 1.0},
        }
    )
    factor_cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["value", "momentum"],
        columns=["value", "momentum"],
    )
    specific_var = pd.Series({"AAA": 0.01, "BBB": 0.04})
    weights = pd.Series({"AAA": 0.6, "BBB": 0.4})
    model = FactorRiskModel(exposures, factor_cov, specific_var)

    exposure = model.portfolio_exposure(weights)
    variance = model.portfolio_variance(weights)
    contribution = model.risk_contribution(weights)

    pd.testing.assert_series_equal(
        exposure,
        pd.Series({"value": 0.4, "momentum": 0.52}, name="exposure"),
    )
    assert math.isclose(variance, 0.044896, rel_tol=1e-12)
    assert math.isclose(model.portfolio_risk(weights), math.sqrt(0.044896), rel_tol=1e-12)
    assert math.isclose(
        contribution["total"].sum(),
        variance,
        rel_tol=1e-12,
    )
    assert math.isclose(contribution.loc["specific", "total"], 0.0100, rel_tol=1e-12)


def test_factor_risk_model_supports_active_risk_against_benchmark() -> None:
    exposures = pd.DataFrame(
        {"value": {"AAA": 1.0, "BBB": -0.5}, "momentum": {"AAA": 0.2, "BBB": 1.0}}
    )
    factor_cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["value", "momentum"],
        columns=["value", "momentum"],
    )
    specific_var = pd.Series({"AAA": 0.01, "BBB": 0.04})
    model = FactorRiskModel(exposures, factor_cov, specific_var)

    active = model.active_risk(
        pd.Series({"AAA": 0.6, "BBB": 0.4}),
        pd.Series({"AAA": 0.5, "BBB": 0.5}),
    )

    assert math.isclose(
        active,
        model.portfolio_risk(pd.Series({"AAA": 0.1, "BBB": -0.1})),
        rel_tol=1e-12,
    )


def test_performance_attribution_splits_common_and_specific_returns() -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    returns = pd.Series([0.0100, -0.0050], index=dates, name="returns")
    positions = pd.DataFrame(
        {"AAA": [0.6, 0.2], "BBB": [0.4, 0.8]},
        index=dates,
    )
    factor_returns = pd.DataFrame(
        {"value": [0.010, -0.020], "momentum": [0.020, 0.010]},
        index=dates,
    )
    factor_loadings = pd.DataFrame(
        {
            "value": [1.0, -0.5, 0.5, 0.0],
            "momentum": [0.2, 1.0, 0.4, 0.8],
        },
        index=pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["date", "symbol"]),
    )
    attribution = PerformanceAttribution(
        returns=returns,
        positions=positions,
        factor_returns=factor_returns,
        factor_loadings=factor_loadings,
    )

    exposures = attribution.exposures()
    frame = attribution.attribution()
    summary, exposure_summary = attribution.summary()

    expected_exposures = pd.DataFrame(
        {"value": [0.4, 0.1], "momentum": [0.52, 0.72]},
        index=dates,
    )
    pd.testing.assert_frame_equal(exposures, expected_exposures)
    pd.testing.assert_series_equal(
        frame["common_returns"],
        frame[["value", "momentum"]].sum(axis=1).rename("common_returns"),
    )
    pd.testing.assert_series_equal(
        frame["specific_returns"],
        (returns - frame["common_returns"]).rename("specific_returns"),
    )
    assert {"common_return_mean", "specific_return_mean", "total_return_mean"}.issubset(summary)
    assert list(exposure_summary.columns) == [
        "average_exposure",
        "cumulative_return_contribution",
    ]
