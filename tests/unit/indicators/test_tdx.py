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


def test_tdx_tdx30_submodule_exposes_classic_functions() -> None:
    """ta.tdx.tdx30 exposes the same classic indicator functions."""
    close = _close()

    result = ta.tdx.tdx30.MA(close, N=4)

    np.testing.assert_allclose(result.to_numpy(), MyTT.MA(close.to_numpy(), 4), equal_nan=True)
    pd.testing.assert_index_equal(result.index, close.index)


def test_tdx_wr_bias_boll_psy_match_mytt_with_stable_columns() -> None:
    """Multi-column tdx oscillators match MyTT formulas."""
    close = _close()
    high = _high()
    low = _low()

    wr = ta.tdx.wr(close, high, low, n=5, n1=3)
    bias = ta.tdx.bias(close, l1=3, l2=5, l3=7)
    boll = ta.tdx.boll(close, n=5, p=2)
    psy = ta.tdx.psy(close, n=5, m=3)

    _assert_frame_matches(
        wr,
        ["WR", "WR1"],
        MyTT.WR(close.to_numpy(), high.to_numpy(), low.to_numpy(), N=5, N1=3),
    )
    _assert_frame_matches(
        bias,
        ["BIAS1", "BIAS2", "BIAS3"],
        MyTT.BIAS(close.to_numpy(), L1=3, L2=5, L3=7),
    )
    _assert_frame_matches(boll, ["UPPER", "MID", "LOWER"], MyTT.BOLL(close.to_numpy(), N=5, P=2))
    _assert_frame_matches(psy, ["PSY", "PSYMA"], MyTT.PSY(close.to_numpy(), N=5, M=3))


def test_tdx_cci_atr_expma_obv_match_mytt() -> None:
    """Single and two-column tdx indicators match MyTT formulas."""
    close = _close()
    high = _high()
    low = _low()
    volume = _volume()

    cci = ta.tdx.cci(close, high, low, n=5)
    atr = ta.tdx.atr(close, high, low, n=5)
    expma = ta.tdx.expma(close, n1=4, n2=8)
    obv = ta.tdx.obv(close, volume)

    np.testing.assert_allclose(
        cci.to_numpy(),
        MyTT.CCI(close.to_numpy(), high.to_numpy(), low.to_numpy(), N=5),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        atr.to_numpy(),
        MyTT.ATR(close.to_numpy(), high.to_numpy(), low.to_numpy(), N=5),
        equal_nan=True,
    )
    _assert_frame_matches(expma, ["EXPMA1", "EXPMA2"], MyTT.EXPMA(close.to_numpy(), N1=4, N2=8))
    np.testing.assert_allclose(
        obv.to_numpy(),
        MyTT.OBV(close.to_numpy(), volume.to_numpy()),
        equal_nan=True,
    )
    pd.testing.assert_index_equal(cci.index, close.index)
    pd.testing.assert_index_equal(atr.index, close.index)
    pd.testing.assert_index_equal(obv.index, close.index)
    assert cci.name == "CCI_5"
    assert atr.name == "ATR_5"
    assert obv.name == "OBV"


def _assert_frame_matches(
    result: pd.DataFrame,
    columns: list[str],
    expected: tuple[np.ndarray, ...],
) -> None:
    """Assert a result frame matches MyTT tuple output."""
    assert list(result.columns) == columns
    for column, values in zip(result.columns, expected, strict=True):
        np.testing.assert_allclose(result[column].to_numpy(), values, equal_nan=True)
    pd.testing.assert_index_equal(result.index, _close().index)


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


def _volume() -> pd.Series:
    """Return deterministic volume values."""
    close = _close()
    return pd.Series(
        [1000, 1200, 900, 1500, 1800, 1100, 1700, 1600, 1300, 1900, 2100, 1400],
        index=close.index,
        name="volume",
    )
