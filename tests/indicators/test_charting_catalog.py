import json

import numpy as np
import pandas as pd
import pytest

from tradelearn.indicators.charting import calculate, catalog


@pytest.fixture
def bars():
    rng = np.random.default_rng(123)
    c = 100 + np.cumsum(rng.normal(0, 1, 260))
    o = c + rng.normal(0, 0.4, len(c))
    return pd.DataFrame(
        dict(
            open=o,
            high=np.maximum(o, c) + 1,
            low=np.minimum(o, c) - 1,
            close=c,
            volume=rng.integers(1, 100, len(c)),
        ),
        index=pd.date_range("2025-01-01", periods=len(c), freq="h", tz="UTC"),
    )


@pytest.mark.parametrize("entry", catalog(), ids=lambda x: x["id"])
def test_catalog_smoke_and_causality(bars, entry):
    result = calculate(bars, entry["id"], secondary=bars)
    json.dumps(result, allow_nan=False)
    assert (
        result.get("profile")
        or result.get("drawings")
        or any(any(v is not None for v in s["values"]) for s in result["series"])
    ), entry["id"]
    assert [s["id"] for s in result["series"]] == [p["id"] for p in entry["plots"]]
    for s in result["series"]:
        assert len(s["values"]) == len(bars)
    prefix = calculate(bars.iloc[:190], entry["id"], secondary=bars.iloc[:190])
    for a, b in zip(result["series"], prefix["series"], strict=True):
        np.testing.assert_allclose(
            np.array(a["values"][:190], dtype=float),
            np.array(b["values"], dtype=float),
            equal_nan=True,
            rtol=1e-9,
            atol=1e-9,
        )


def test_numerical_formulas(bars):
    def values(name, params=None):
        return np.array(calculate(bars, name, params)["series"][0]["values"], dtype=float)

    np.testing.assert_allclose(
        values("Moving Average", {"length": 5}), bars.close.rolling(5).mean(), equal_nan=True
    )
    np.testing.assert_allclose(
        values("Advance/Decline", {"length": 9}),
        (bars.close > bars.open).rolling(9).sum()
        / (bars.close < bars.open).rolling(9).sum().replace(0, np.nan),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        values("Price Volume Trend"),
        (bars.close.pct_change().fillna(0) * bars.volume).cumsum(),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        values("Linear Regression Slope", {"length": 5}),
        bars.close.rolling(5).apply(lambda x: np.polyfit(np.arange(5), x, 1)[0], raw=True),
        equal_nan=True,
        atol=1e-12,
    )


def test_profiles_conserve_volume_and_filter(bars):
    result = calculate(bars, "Volume Profile Fixed Range", {"bars": 50, "bins": 17})
    assert sum(b["volume"] for b in result["profile"]) == pytest.approx(bars.volume.tail(50).sum())
    result = calculate(
        bars,
        "Volume Profile Visible Range",
        visible_range={"from": bars.index[30].timestamp(), "to": bars.index[70].timestamp()},
    )
    assert sum(b["volume"] for b in result["profile"]) == pytest.approx(
        bars.volume.iloc[30:71].sum()
    )


def test_secondary_alignment(bars):
    result = calculate(bars, "Spread", secondary=bars.iloc[::2])["series"][0]["values"]
    assert result[1] is None and result[0] == 0
    with pytest.raises(ValueError):
        calculate(bars, "Ratio")


def test_validation_and_isolation(bars):
    assert len(catalog()) == 105
    copy = catalog()
    copy[0]["params"].append({})
    assert catalog() != copy
    with pytest.raises(ValueError):
        calculate(bars, "Moving Average", {"length": 0})
    with pytest.raises(ValueError):
        calculate(bars, "Moving Average", {"unused": 1})
    with pytest.raises(ValueError):
        calculate(bars.iloc[::-1], "Moving Average")


def test_vwap_anchor_and_pivot_previous_day(bars):
    values = calculate(bars, "VWAP")["series"][0]["values"]
    typical = (bars.high + bars.low + bars.close) / 3
    assert values[24] == pytest.approx(typical.iloc[24])
    result = calculate(bars, "Pivot Points Standard")["series"][0]["values"]
    assert result[23] is None
    assert result[24] == pytest.approx(
        (bars.high.iloc[:24].max() + bars.low.iloc[:24].min() + bars.close.iloc[23]) / 3
    )


def test_advance_decline_without_declining_bars_is_undefined(bars):
    bars["open"] = bars.close - 0.1
    values = calculate(bars, "Advance/Decline", {"length": 3})["series"][0]["values"]
    assert all(value is None for value in values)
