"""Regression cases derived from bundled Charting study constructors."""

import numpy as np
import pandas as pd
import pytest

from tradelearn.indicators import tv
from tradelearn.indicators.charting import calculate


def frame():
    x = np.arange(100, dtype=float)
    close = 50 + x * 0.2 + 3 * np.sin(x * 0.73)
    width = 1 + (x % 7)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + width,
            "low": close - width / 2,
            "close": close,
            "volume": 100 + x,
        },
        index=pd.date_range("2025-01-01", periods=len(x), tz="UTC"),
    )


def series(f, name, params):
    return {
        item["id"]: np.asarray(item["values"], dtype=float)
        for item in calculate(f, name, params)["series"]
    }


def test_trix_uses_log_prices_and_basis_point_changes():
    f = frame()
    smooth = np.log(f.close)
    for _ in range(3):
        smooth = tv.ema(smooth, 4)
    expected = smooth.diff() * 10000
    np.testing.assert_allclose(series(f, "TRIX", {"length": 4})["value"], expected, equal_nan=True)


def test_ma_with_ema_cross_keeps_each_period_with_its_method():
    f = frame()
    actual = series(f, "MA with EMA Cross", {"fast": 3, "slow": 8})
    np.testing.assert_allclose(actual["fast"], tv.sma(f.close, 3), equal_nan=True)
    np.testing.assert_allclose(actual["slow"], tv.ema(f.close, 8), equal_nan=True)


def test_keltner_range_uses_ema_not_wilder_smoothing():
    f = frame()
    ranges = pd.concat(
        [f.high - f.low, (f.high - f.close.shift()).abs(), (f.low - f.close.shift()).abs()], axis=1
    ).max(axis=1)
    actual = series(f, "Keltner Channels", {"length": 5, "mult": 1.7})
    np.testing.assert_allclose(
        actual["upper"] - actual["mid"], tv.ema(ranges, 5) * 1.7, equal_nan=True
    )


@pytest.mark.parametrize("length", [7, 14])
def test_hull_uses_nearest_integer_square_root(length):
    f = frame()
    expected = tv.wma(
        2 * tv.wma(f.close, length // 2) - tv.wma(f.close, length),
        int(np.floor(np.sqrt(length) + 0.5)),
    )
    actual = series(f, "Hull Moving Average", {"length": length})
    np.testing.assert_allclose(actual["value"], expected, equal_nan=True)


def test_hull_length_one_is_supported_by_catalog():
    f = frame()
    actual = series(f, "Hull Moving Average", {"length": 1})
    np.testing.assert_allclose(actual["value"], f.close)


@pytest.mark.parametrize("rank_length", [1, 4])
def test_connors_rank_counts_ties_and_divides_by_full_window(rank_length):
    f = frame()
    # Equal adjacent returns distinguish <= from < in the original percentrank.
    close = pd.Series(np.resize([10.0, 10.0, 10.0, 12.0, 10.0], len(f)), index=f.index)
    f["open"] = f["close"] = close
    f["high"] = close + 1
    f["low"] = close - 1
    streak = []
    last = 0
    for change in close.diff():
        last = max(last, 0) + 1 if change > 0 else min(last, 0) - 1 if change < 0 else 0
        streak.append(last)
    rank = (
        close.pct_change(fill_method=None)
        .rolling(rank_length)
        .apply(
            lambda values: np.count_nonzero(values[:-1] <= values[-1]) * 100 / rank_length, raw=True
        )
    )
    expected = (
        tv.rsi(close, 3) + tv.rsi(pd.Series(streak, index=f.index, dtype=float), 2) + rank
    ) / 3
    actual = series(
        f, "Connors RSI", {"rsi_length": 3, "streak_length": 2, "rank_length": rank_length}
    )
    np.testing.assert_allclose(actual["value"], expected, equal_nan=True)


def test_fractals_do_not_signal_in_a_flat_market():
    f = frame()
    f[["open", "high", "low", "close"]] = 10.0
    actual = series(f, "Williams Fractal", {"period": 2})
    assert np.isnan(actual["high"]).all()
    assert np.isnan(actual["low"]).all()


def test_fractal_plateau_requires_strictly_lower_newer_highs():
    f = frame().iloc[:7].copy()
    f["high"] = [1.0, 2.0, 3.0, 3.0, 3.0, 2.0, 1.0]
    f["open"] = f["close"] = f.high - 0.5
    f["low"] = 0.0
    actual = series(f, "Williams Fractal", {"period": 2})
    # Confirm the last plateau high, with two strictly lower newer bars.
    np.testing.assert_allclose(actual["high"], [np.nan] * 6 + [3.0], equal_nan=True)
    assert np.isnan(actual["low"]).all()


def test_dmi_adxr_uses_length_minus_one_lag():
    frame = pd.DataFrame(
        {
            "open": [100 + i % 5 for i in range(80)],
            "high": [102 + i % 5 for i in range(80)],
            "low": [99 + i % 5 for i in range(80)],
            "close": [101 + i % 5 for i in range(80)],
            "volume": 100,
        },
        index=pd.date_range("2025-01-01", periods=80, freq="h", tz="UTC"),
    )
    result = calculate(frame, "Directional Movement", {"length": 5, "adx_smoothing": 7})
    values = {s["id"]: pd.Series(s["values"], dtype=float) for s in result["series"]}
    np.testing.assert_allclose(
        values["adxr"], (values["adx"] + values["adx"].shift(4)) / 2, equal_nan=True
    )
