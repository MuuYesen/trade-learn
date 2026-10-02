"""Boundary and alignment contracts for factor cleaning and event metrics."""

import numpy as np
import pandas as pd
import pytest

from tradelearn.metrics.factor import (
    clean_factor_and_forward_returns,
    event_returns,
    mean_monthly_ic,
    quantile_returns,
)


def _inputs():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=3), ["A", "B"]], names=["date", "symbol"]
    )
    return (
        pd.DataFrame(
            {"value": [1.0, 2.0] * 3, "other": [2.0, 1.0] * 3, "sector": ["tech", "bank"] * 3},
            index=index,
        ),
        pd.Series([100.0, 100.0, 110.0, 90.0, 121.0, 81.0], index=index),
    )


@pytest.mark.parametrize("price_form", ["series", "close", "single_column", "bars_index"])
def test_clean_factors_preserves_forward_alignment_across_input_forms(price_form):
    factors, prices = _inputs()
    if price_form == "close":
        prices = prices.to_frame("close")
    elif price_form == "single_column":
        prices = prices.to_frame("adjusted_close")
    elif price_form == "bars_index":
        prices.index = prices.index.set_names(["timestamp", "symbol"])
    clean = clean_factor_and_forward_returns(
        factors.reset_index(),
        factor=["value", "other"],
        prices=prices,
        periods=(1,),
        quantiles=2,
        groupby="sector",
    )
    assert len(clean) == 8
    assert set(clean["factor_name"]) == {"value", "other"}
    assert set(clean["group"]) == {"tech", "bank"}
    assert clean.xs("A", level="symbol")["forward_return_1"].tolist() == pytest.approx([0.1] * 4)
    assert clean.xs("B", level="symbol")["forward_return_1"].tolist() == pytest.approx([-0.1] * 4)


@pytest.mark.parametrize(
    "changes, error, message",
    [
        ({"quantiles": 0}, ValueError, "quantiles"),
        ({"factor": []}, ValueError, "at least one"),
        ({"factor": "absent"}, ValueError, "factor column"),
        ({"factor": "value", "periods": (0,)}, ValueError, "periods"),
        ({"factor": ["value", "other"], "periods": (-1,)}, ValueError, "periods"),
        ({"prices": [1, 2]}, TypeError, "prices"),
        ({"prices": pd.DataFrame({"a": [1], "b": [2]})}, ValueError, "close"),
        ({"groupby": "absent"}, ValueError, "groupby"),
    ],
)
def test_clean_factors_rejects_invalid_public_inputs(changes, error, message):
    factors, prices = _inputs()
    kwargs = {"factor": "value", "prices": prices, "quantiles": 2, **changes}
    with pytest.raises(error, match=message):
        clean_factor_and_forward_returns(factors, **kwargs)


def test_clean_factors_rejects_non_tables_and_non_multiindex():
    factors, prices = _inputs()
    with pytest.raises(TypeError, match="DataFrame"):
        clean_factor_and_forward_returns(factors["value"], factor="value", prices=prices)
    with pytest.raises(ValueError, match="MultiIndex"):
        clean_factor_and_forward_returns(
            factors.reset_index(drop=True), factor="value", prices=prices
        )


def test_clean_empty_factors_keeps_an_empty_quantile_contract():
    factors, prices = _inputs()
    clean = clean_factor_and_forward_returns(factors.iloc[:0], factor="value", prices=prices)
    assert clean.empty
    assert "factor_quantile" in clean


def test_monthly_ic_averages_available_factor_columns_before_months():
    frame = pd.DataFrame(
        {"a": [0.1, 0.3, np.nan, -0.2], "b": [0.3, 0.5, np.nan, -0.4]},
        index=pd.to_datetime(["2024-01-01", "2024-01-03", "2024-02-01", "2025-02-01"]),
    )
    result = mean_monthly_ic(frame)
    assert result.loc[2024, 1] == pytest.approx(0.3)
    assert result.loc[2025, 2] == pytest.approx(-0.3)
    assert np.isnan(result.loc[2024, 2])
    assert result.index.name == "year" and result.columns.name == "month"


def test_event_windows_skip_unavailable_events_without_contaminating_valid_windows():
    _, prices = _inputs()
    events = pd.MultiIndex.from_tuples(
        [
            ("2024-01-02", "A"),
            ("2024-01-02", "B"),
            ("2024-01-02", "missing"),
            ("1999-01-01", "A"),
            ("2024-01-01", "A"),
        ]
    )
    result = event_returns(prices, events, before=1, after=1)
    assert result["count"].tolist() == [2, 2, 2]
    assert result["mean"].tolist() == pytest.approx([1 / 99, 0.0, 0.0])
    prices.loc[(pd.Timestamp("2024-01-02"), "A")] = 0
    prices.loc[(pd.Timestamp("2024-01-02"), "B")] = np.nan
    assert event_returns(prices, events, before=1, after=1).isna().all().all()
    with pytest.raises(ValueError, match="non-negative"):
        event_returns(prices, events, before=-1)
    with pytest.raises(ValueError, match="events must"):
        event_returns(prices, pd.Index(["2024-01-02"]))


def test_group_neutral_quantiles_require_groups_and_remove_group_mean():
    factors, prices = _inputs()
    with pytest.raises(ValueError, match="requires groupby"):
        quantile_returns(factors["value"], prices, quantiles=2, group_neutral=True)
    result = quantile_returns(
        factors["value"], prices, quantiles=2, groupby=factors["sector"], group_neutral=True
    )
    assert (result == 0).all().all()
    with pytest.raises(ValueError, match="quantiles"):
        quantile_returns(factors["value"], prices, quantiles=0)


def test_monthly_ic_accepts_a_single_ic_series():
    values = pd.Series(
        [0.1, np.nan, 0.3, -0.2],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-02-01"]),
    )
    result = mean_monthly_ic(values)
    assert result.loc[2024, 1] == pytest.approx(0.2)
    assert result.loc[2024, 2] == pytest.approx(-0.2)


def test_clean_factors_aligns_external_groups_and_drops_unidentifiable_tied_bins():
    factors, prices = _inputs()
    groups = factors["sector"].iloc[::-1]
    clean = clean_factor_and_forward_returns(
        factors,
        factor="value",
        prices=prices,
        quantiles=2,
        groupby=groups,
    )
    assert clean.xs("A", level="symbol")["group"].tolist() == ["tech", "tech"]
    assert clean.xs("B", level="symbol")["group"].tolist() == ["bank", "bank"]
    factors["value"] = 1.0
    tied = clean_factor_and_forward_returns(
        factors,
        factor="value",
        prices=prices,
        quantiles=2,
    )
    assert tied.empty
    assert "factor_quantile" in tied


def test_clean_factors_does_not_suppress_invalid_quantile_errors():
    factors, prices = _inputs()
    with pytest.raises(ValueError, match="percentiles"):
        clean_factor_and_forward_returns(
            factors,
            factor="value",
            prices=prices,
            quantiles=float("nan"),
        )
