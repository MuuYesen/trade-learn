"""Anchors, causal pivots, missing input and index counterexamples."""

import sys

import numpy as np
import pandas as pd
import pytest

if sys.version_info < (3, 11):
    pytest.skip(
        "Fixed PyneCore 6.13.3 requires Python 3.11+; TV is unsupported on 3.10",
        allow_module_level=True,
    )

from tradelearn.indicators import tv


def inputs():
    x = pd.Series([10.0, 20.0, 30.0, 40.0])
    volume = pd.Series([1.0, 2.0, 1.0, 1.0])
    anchor = pd.Series([True, False, True, False])
    return x, volume, anchor


def test_explicit_source_volume_and_anchor_vwap():
    x, v, a = inputs()
    np.testing.assert_allclose(tv.vwap(x, v, a), [10, 50 / 3, 30, 35])
    result = tv.vwap(x, v, a, stdev_mult=2.0)
    assert result.shape == (4, 3)
    assert result.iloc[0, 0] == 10


def test_anchor_requires_nonmissing_boolean_values():
    x, v, a = inputs()
    for invalid in [pd.Series([np.nan, 0, 1, 0]), pd.Series([2, 0, 1, 0])]:
        with pytest.raises(ValueError, match="booleans"):
            tv.vwap(x, v, invalid)


def test_inputs_require_exact_index_alignment():
    x, v, a = inputs()
    with pytest.raises(ValueError, match="indexes"):
        tv.vwap(x, v.set_axis([4, 5, 6, 7]), a)
    with pytest.raises(ValueError, match="indexes"):
        tv.correlation(x, x.set_axis([4, 5, 6, 7]))


def test_pivots_only_change_after_period_confirmation():
    x, v, a = inputs()
    actual = tv.pivot_point_levels(x, x + 1, x - 1, x, a)
    assert actual.iloc[:2].isna().all().all()
    assert actual.iloc[2, 0] == pytest.approx((21 + 9 + 20) / 3)
    pd.testing.assert_frame_equal(
        actual.iloc[:3],
        tv.pivot_point_levels(
            x.iloc[:3], (x + 1).iloc[:3], (x - 1).iloc[:3], x.iloc[:3], a.iloc[:3]
        ),
    )


def test_empty_numeric_outputs_and_upstream_defined_degenerate_correlation():
    empty = pd.Series(dtype=float)
    assert tv.sma(empty).empty
    result = tv.correlation(pd.Series([1.0] * 4), pd.Series([1.0] * 4), length=2)
    assert pd.api.types.is_numeric_dtype(result)
    # Fixed upstream returns zero for a defined zero-variance pair after warmup.
    assert result.iloc[1:].eq(0).all()


def test_intraday_intensity_requires_the_actual_close_input():
    x = pd.Series([10.0, 11.0, 12.0])
    high = x + 2
    low = x - 1
    volume = pd.Series([100.0, 200.0, 300.0])
    result = tv.iii(high, low, x, volume)
    np.testing.assert_allclose(result, (2 * x - high - low) / ((high - low) * volume))


@pytest.mark.parametrize(
    "name,params,expected",
    [
        ("atr", {"length": 7}, 7),
        ("cmo", {"length": 7}, 8),
        ("hma", {"length": 20}, 23),
        ("dmi", {"diLength": 7, "adxSmoothing": 4}, 11),
        ("kc", {"length": 7}, 8),
        ("tsi", {"short_length": 5, "long_length": 9}, 14),
        ("swma", {}, 4),
        ("percentrank", {"length": 7}, 8),
        ("stoch", {"length": 7}, 7),
    ],
)
def test_source_warmup_metadata(name, params, expected):
    assert getattr(tv, name).configured(**params).minimum_period == expected
