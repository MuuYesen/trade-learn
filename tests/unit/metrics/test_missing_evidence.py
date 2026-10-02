"""Missing observations must not silently become complete financial evidence."""

import math

import numpy as np
import pandas as pd
import pytest

from tradelearn import metrics


def test_simple_returns_does_not_fill_missing_prices():
    result = metrics.simple_returns(pd.Series([100.0, np.nan, 110.0]))
    assert result.isna().all()


def test_compounding_propagates_a_gap_to_later_values():
    result = metrics.cum_returns(pd.Series([0.1, np.nan, 0.2]), nan_policy="propagate")
    assert result.iloc[0] == pytest.approx(0.1)
    assert result.iloc[1:].isna().all()


@pytest.mark.parametrize(
    "name", ["annual_return", "volatility", "sharpe", "sortino", "downside_risk", "calmar", "ic_ir"]
)
def test_scalar_metrics_propagate_missing_period(name):
    assert math.isnan(
        getattr(metrics, name)(pd.Series([0.1, np.nan, -0.05]), periods=252, nan_policy="propagate")
    )


@pytest.mark.parametrize("name", ["max_drawdown", "omega", "var", "cvar", "tail_ratio"])
def test_scalar_metrics_without_required_period_propagate_missing(name):
    assert math.isnan(
        getattr(metrics, name)(pd.Series([0.1, np.nan, -0.05]), nan_policy="propagate")
    )


@pytest.mark.parametrize("name", ["beta", "alpha", "information_ratio"])
def test_pair_metrics_do_not_drop_nan_when_propagation_requested(name):
    args = {} if name == "beta" else {"periods": 252}
    assert math.isnan(
        getattr(metrics, name)(
            pd.Series([0.1, np.nan, -0.05]),
            pd.Series([0.02, 0.03, -0.01]),
            nan_policy="propagate",
            **args,
        )
    )


def test_annual_dataframe_propagates_per_column():
    result = metrics.annual_return(
        pd.DataFrame({"missing": [0.1, np.nan], "complete": [0.1, 0.2]}),
        periods=2,
        nan_policy="propagate",
    )
    assert math.isnan(result["missing"])
    assert result["complete"] == pytest.approx(0.32)


@pytest.mark.parametrize("periods", [True, 1.5, float("nan"), float("inf")])
def test_annualization_requires_actual_positive_integer(periods):
    with pytest.raises(ValueError, match="periods"):
        metrics.annual_return(pd.Series([0.1, 0.2]), periods=periods)


@pytest.mark.parametrize("name", ["annual_return", "sharpe", "volatility"])
def test_nonfinite_observation_rejected(name):
    with pytest.raises(ValueError, match="infinite"):
        getattr(metrics, name)(pd.Series([0.1, np.inf]), periods=252)


@pytest.mark.parametrize("name", ["var", "cvar", "tail_ratio"])
def test_no_observations_returns_unavailable_not_index_error(name):
    assert math.isnan(getattr(metrics, name)(pd.Series([], dtype=float)))


@pytest.mark.parametrize("name", ["ic", "rank_ic"])
@pytest.mark.parametrize("by_group", [False, True])
def test_factor_correlations_propagate_missing_pairs(name, by_group):
    index = pd.MultiIndex.from_product([[pd.Timestamp("2026-09-21")], ["A", "B", "C"]])
    factor = pd.Series([1.0, 2.0, 3.0], index=index)
    forward = pd.Series([0.1, np.nan, 0.3], index=index)
    options = {"groupby": pd.Series("sector", index=index), "by_group": True} if by_group else {}
    result = getattr(metrics, name)(factor, forward, nan_policy="propagate", **options)
    assert np.isnan(result.to_numpy()).all()


@pytest.mark.parametrize("name", ["autocorrelation", "turnover"])
def test_rank_history_propagates_missing_observation(name):
    index = pd.MultiIndex.from_product([[1, 2], ["A", "B", "C"]])
    factor = pd.Series([1.0, 2.0, 3.0, 1.0, np.nan, 3.0], index=index)
    assert getattr(metrics, name)(factor, nan_policy="propagate").isna().all()


@pytest.mark.parametrize("missing_factor", [False, True])
def test_quantile_mean_does_not_hide_missing_evidence(missing_factor):
    index = pd.MultiIndex.from_product([[1], ["A", "B", "C", "D"]])
    factor = pd.Series([1.0, np.nan if missing_factor else 2.0, 3.0, 4.0], index=index)
    forward = pd.Series([0.1, np.nan if not missing_factor else 0.2, 0.3, 0.4], index=index)
    result = metrics.quantile_returns(factor, forward, quantiles=2, nan_policy="propagate")
    assert result.iloc[:, 0].isna().all()


def test_factor_returns_does_not_fill_missing_prices():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=3), ["A", "B"]],
        names=["date", "symbol"],
    )
    prices = pd.Series([100.0, 100.0, np.nan, 100.0, 110.0, 100.0], index=index)
    factor = pd.Series([1.0, 2.0, 1.0, 2.0], index=index[:4])
    result = metrics.factor_returns(factor, prices, quantiles=2, nan_policy="propagate")
    assert result[1].isna().all()
    assert result[2].eq(0).all()
