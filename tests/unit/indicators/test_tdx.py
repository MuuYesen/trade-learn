"""Tests for Tongdaxin-style indicator wrappers."""

import numpy as np
import pandas as pd

from tradelearn import ta
from tradelearn.query.tec import MyTT


def test_tdx_ma_and_ema_match_mytt_and_preserve_index() -> None:
    """ta.tdx moving averages match the vendored MyTT formulas."""
    close = _close()

    ma = ta.tdx.ma(close, n=5)
    ema = ta.tdx.EMA(close, N=6)

    np.testing.assert_allclose(ma.to_numpy(), MyTT.MA(close.to_numpy(), 5), equal_nan=True)
    np.testing.assert_allclose(ema.to_numpy(), MyTT.EMA(close.to_numpy(), 6), equal_nan=True)
    pd.testing.assert_index_equal(ma.index, close.index)
    pd.testing.assert_index_equal(ema.index, close.index)
    assert ma.name == "MA_5"
    assert ema.name == "EMA_6"


def test_tdx_macd_matches_mytt_with_stable_columns() -> None:
    """ta.tdx.macd matches MyTT MACD output and returns named columns."""
    close = _close()

    result = ta.tdx.macd(close, short=4, long=9, m=3)
    expected = MyTT.MACD(close.to_numpy(), SHORT=4, LONG=9, M=3)

    assert list(result.columns) == ["DIF", "DEA", "MACD"]
    for column, values in zip(result.columns, expected, strict=True):
        np.testing.assert_allclose(result[column].to_numpy(), values, equal_nan=True)
    pd.testing.assert_index_equal(result.index, close.index)


def test_tdx_kdj_matches_mytt_with_stable_columns() -> None:
    """ta.tdx.kdj matches MyTT KDJ output and returns named columns."""
    close = _close()
    high = _high()
    low = _low()

    result = ta.tdx.kdj(close, high, low, n=5, m1=3, m2=3)
    expected = MyTT.KDJ(close.to_numpy(), high.to_numpy(), low.to_numpy(), N=5, M1=3, M2=3)

    assert list(result.columns) == ["K", "D", "J"]
    for column, values in zip(result.columns, expected, strict=True):
        np.testing.assert_allclose(result[column].to_numpy(), values, equal_nan=True)
    pd.testing.assert_index_equal(result.index, close.index)


def test_tdx_rsi_matches_mytt_and_tdx30_alias() -> None:
    """ta.tdx.rsi matches MyTT RSI and ta.tdx30 aliases the same namespace."""
    close = _close()

    result = ta.tdx.rsi(close, n=6)

    np.testing.assert_allclose(result.to_numpy(), MyTT.RSI(close.to_numpy(), N=6), equal_nan=True)
    pd.testing.assert_index_equal(result.index, close.index)
    assert result.name == "RSI_6"
    assert ta.tdx30 is ta.tdx


def _close() -> pd.Series:
    """Return deterministic close prices with a date index."""
    return pd.Series(
        [
            10.0,
            10.5,
            10.2,
            11.0,
            11.5,
            11.2,
            12.0,
            12.4,
            12.1,
            12.8,
            13.2,
            13.0,
        ],
        index=pd.date_range("2024-01-01", periods=12, freq="D"),
        name="close",
    )


def _high() -> pd.Series:
    """Return deterministic high prices."""
    close = _close()
    return pd.Series(close.to_numpy() + 0.4, index=close.index, name="high")


def _low() -> pd.Series:
    """Return deterministic low prices."""
    close = _close()
    return pd.Series(close.to_numpy() - 0.3, index=close.index, name="low")
