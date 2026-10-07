"""Numerical and input-contract tests for the extended native wrappers."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from tradelearn import ta


def bars():
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 20.0, 21.0],
            "high": [12.0, 14.0, 23.0, 24.0],
            "low": [8.0, 10.0, 19.0, 20.0],
            "close": [11.0, 13.0, 22.0, 23.0],
            "volume": [2.0, 6.0, 3.0, 1.0],
        },
        index=pd.date_range("2024-01-01", periods=4),
    )


def test_native_volume_and_range_formulas():
    f = bars()
    np.testing.assert_allclose(ta.tv.accdist(f.high, f.low, f.close, f.volume), [1, 4, 5.5, 6])
    expected = (f.close.pct_change() * f.volume).cumsum()
    np.testing.assert_allclose(ta.tv.pvt(f.close, f.volume), expected, equal_nan=True)
    np.testing.assert_allclose(
        ta.tv.vwma(f.close, f.volume, length=2),
        (f.close * f.volume).rolling(2).sum() / f.volume.rolling(2).sum(),
        equal_nan=True,
    )
    expected = (
        -100
        * (f.high.rolling(2).max() - f.close)
        / (f.high.rolling(2).max() - f.low.rolling(2).min())
    )
    np.testing.assert_allclose(
        ta.tv.wpr(f.high, f.low, f.close, length=2), expected, equal_nan=True
    )


def test_correlation_and_aliases():
    s = pd.Series([2.0, 5.0, 1.0, 8.0, 3.0])
    np.testing.assert_allclose(ta.tv.correlation(s, -2 * s, length=3).iloc[2:], -1)
    for name in ("accdist", "pvt", "wpr", "vwma", "correlation", "pivot_point_levels"):
        assert getattr(ta.tv, name.upper()) is getattr(ta.tv, name)
        assert name in ta.tv.COVERED_PYNECORE


def test_vwap_source_anchor_and_weighted_bands():
    f = bars()
    anchor = pd.Series([False, True, False, True], index=f.index)
    result = ta.tv.vwap(
        f.high, f.low, f.close, f.volume, source=f.open, anchor=anchor, stdev_mult=2
    )
    assert list(result) == ["vwap", "upper", "lower"]
    assert result.iloc[0].isna().all()
    np.testing.assert_allclose(result.vwap.iloc[1:], [11, 14, 21])
    np.testing.assert_allclose(result.upper.iloc[1:], [11, 14 + 2 * np.sqrt(18), 21])
    np.testing.assert_allclose(result.lower.iloc[1:], [11, 14 - 2 * np.sqrt(18), 21])
    default = ta.tv.vwap(f.high, f.low, f.close, f.volume)
    np.testing.assert_allclose(default, (f.close * f.volume).cumsum() / f.volume.cumsum())


def test_pivots_previous_period_open_and_causality():
    f = bars()
    anchor = pd.Series([True, False, True, False], index=f.index)
    result = ta.tv.pivot_point_levels(f.open, f.high, f.low, f.close, anchor)
    assert result.iloc[:2].isna().all().all()
    np.testing.assert_allclose(result.p.iloc[2:], (14 + 8 + 13) / 3)
    assert result[["r4", "s4", "r5", "s5"]].isna().all().all()
    woodie = ta.tv.pivot_point_levels(f.open, f.high, f.low, f.close, anchor, type="Woodie")
    np.testing.assert_allclose(woodie.p.iloc[2:], (14 + 8 + 2 * 20) / 4)
    prefix = ta.tv.pivot_point_levels(
        f.open.iloc[:3], f.high.iloc[:3], f.low.iloc[:3], f.close.iloc[:3], anchor.iloc[:3]
    )
    assert_frame_equal(result.iloc[:3], prefix)


@pytest.mark.parametrize(
    "operation",
    [
        lambda f, bad: ta.tv.atr(bad, f.low, f.close),
        lambda f, bad: ta.tv.vwma(f.close, bad),
        lambda f, bad: ta.tv.correlation(f.close, bad),
        lambda f, bad: ta.tv.vwap(f.high, f.low, f.close, f.volume, source=bad),
        lambda f, bad: ta.tv.vwap(f.high, f.low, f.close, f.volume, anchor=bad.astype(bool)),
    ],
)
def test_inputs_require_exact_index_alignment(operation):
    f = bars()
    with pytest.raises(ValueError, match="index"):
        operation(f, f.close.reset_index(drop=True))


def test_anchor_requires_nonmissing_booleans():
    f = bars()
    for anchor in (f.close, pd.Series([True, False, None, True], index=f.index)):
        with pytest.raises(ValueError, match="bool"):
            ta.tv.vwap(f.high, f.low, f.close, f.volume, anchor=anchor)


def test_concurrent_calls_are_isolated():
    inputs = [pd.Series(np.arange(500.0) + i) for i in range(12)]
    expected = [ta.tv.ema(s, length=7) for s in inputs]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda s: ta.tv.ema(s, length=7), inputs))
    for left, right in zip(expected, actual, strict=True):
        assert_series_equal(left, right)


def test_empty_and_undefined_results_remain_numeric():
    empty = pd.Series(dtype=float)
    assert ta.tv.vwap(empty, empty, empty, empty).empty
    assert ta.tv.vwap(empty, empty, empty, empty, stdev_mult=2).shape == (0, 3)
    s = pd.Series([1.0, 2.0, 3.0])
    result = ta.tv.correlation(s, s * 0, length=2)
    assert result.isna().all()
    assert pd.api.types.is_float_dtype(result.dtype)


def test_vwap_missing_values_and_zero_volume_bands():
    f = bars()
    source = f.close.copy()
    source.iloc[1] = np.nan
    result = ta.tv.vwap(f.high, f.low, f.close, f.volume, source=source, stdev_mult=2)
    assert result.iloc[1].isna().all()
    assert result.vwap.iloc[2] == pytest.approx((11 * 2 + 22 * 3) / 5)
    zero = ta.tv.vwap(f.high, f.low, f.close, f.volume * 0, stdev_mult=2)
    assert zero.isna().all().all()
