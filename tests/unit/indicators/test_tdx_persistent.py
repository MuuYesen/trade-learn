"""Explicit numerical migration from old TL approximations to fixed MyTT."""

import numpy as np
import pandas as pd

from tradelearn.indicators import tdx
from tradelearn.indicators.tdx.mytt_adapter import MyTT as legacy


def test_fixed_mytt_has_full_window_warmup_instead_of_partial_average():
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    assert tdx.MA(close, N=5).iloc[:4].isna().all()
    assert legacy.MA(close, 5)[0] == 10
    assert tdx.MA.minimum_period == 5


def test_obv_retains_upstream_ten_thousand_unit_divisor():
    close = pd.Series([10.0, 11.0, 12.0, 11.0])
    volume = pd.Series([10000.0] * 4)
    np.testing.assert_allclose(tdx.OBV(close, volume), [0.0, 1.0, 2.0, 1.0])
    np.testing.assert_allclose(legacy.OBV(close, volume), [0.0, 10000.0, 20000.0, 10000.0])


def test_fixed_macd_uses_upstream_rounding_and_preserves_output_order():
    close = pd.Series(np.linspace(10, 30, 100) + np.sin(np.arange(100)))
    result = tdx.MACD(close)
    assert list(result) == ["DIF", "DEA", "MACD"]
    np.testing.assert_array_equal(result.to_numpy(), np.round(result.to_numpy(), 3))


def test_ktn_source_uses_typical_price_and_two_atr_bands():
    close = pd.Series(np.arange(40, dtype=float) + 20)
    high, low = close + 2, close - 1
    result = tdx.KTN(close, high, low, N=5, M=3)
    atr = tdx.ATR(close, high, low, N=3)
    np.testing.assert_allclose(result.UPPER - result.MID, atr * 2, equal_nan=True)
    assert not np.allclose(result.UPPER.iloc[10:], legacy.KTN(close, high, low, N=5, M=3)[0][10:])
