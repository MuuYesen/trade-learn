"""Tests for generic indicator wrappers."""

import pandas as pd
import pandas_ta_classic as pta

from tradelearn import ta


def test_sma_matches_pandas_ta_classic() -> None:
    """ta.sma delegates to pandas-ta-classic SMA."""
    close = _close()

    result = ta.sma(close, period=3)

    pd.testing.assert_series_equal(result, pta.sma(close, length=3))


def test_ema_matches_pandas_ta_classic() -> None:
    """ta.ema delegates to pandas-ta-classic EMA."""
    close = _close()

    result = ta.ema(close, length=4)

    pd.testing.assert_series_equal(result, pta.ema(close, length=4))


def test_rsi_matches_pandas_ta_classic() -> None:
    """ta.rsi delegates to pandas-ta-classic RSI."""
    close = _close()

    result = ta.rsi(close, length=4)

    pd.testing.assert_series_equal(result, pta.rsi(close, length=4))


def test_macd_matches_pandas_ta_classic_with_stable_column_names() -> None:
    """ta.macd delegates to pandas-ta-classic and normalizes output columns."""
    close = _close()

    result = ta.macd(close, fast=3, slow=6, signal=2)
    expected = pta.macd(close, fast=3, slow=6, signal=2)
    expected.columns = ["macd", "hist", "signal"]

    pd.testing.assert_frame_equal(result, expected)


def test_bbands_matches_pandas_ta_classic_with_stable_column_names() -> None:
    """ta.bbands delegates to pandas-ta-classic and normalizes output columns."""
    close = _close()

    result = ta.bbands(close, length=4, std=2)
    expected = pta.bbands(close, length=4, std=2)
    expected = expected.iloc[:, :3].copy()
    expected.columns = ["lower", "mid", "upper"]

    pd.testing.assert_frame_equal(result, expected)


def test_atr_adx_vwap_match_pandas_ta_classic() -> None:
    """Common OHLCV indicators delegate to pandas-ta-classic."""
    frame = _ohlcv()

    pd.testing.assert_series_equal(
        ta.atr(frame.high, frame.low, frame.close, length=5),
        pta.atr(frame.high, frame.low, frame.close, length=5),
    )

    adx = ta.adx(frame.high, frame.low, frame.close, length=5)
    expected_adx = pta.adx(frame.high, frame.low, frame.close, length=5)
    expected_adx.columns = ["adx", "dmp", "dmn"]
    pd.testing.assert_frame_equal(adx, expected_adx)

    pd.testing.assert_series_equal(
        ta.vwap(frame.high, frame.low, frame.close, frame.volume),
        pta.vwap(frame.high, frame.low, frame.close, frame.volume),
    )


def test_indicator_wrapper_exposes_compute_and_on_bar_error() -> None:
    """Function-style indicators also expose the Indicator shape."""
    close = _close()

    result = ta.sma.compute(close, period=3)

    pd.testing.assert_series_equal(result, ta.sma(close, period=3))
    try:
        ta.sma.on_bar(object())
    except NotImplementedError as exc:
        assert "streaming" in str(exc)
    else:
        raise AssertionError("on_bar should raise until streaming mode is implemented")


def _close() -> pd.Series:
    """Return a deterministic close series."""
    return pd.Series(
        [10.0, 11.0, 10.5, 12.0, 13.0, 12.5, 14.0, 15.0, 14.5, 16.0],
        name="close",
    )


def _ohlcv() -> pd.DataFrame:
    """Return deterministic daily OHLCV bars."""
    index = pd.date_range("2024-01-01", periods=40, freq="D")
    return pd.DataFrame(
        {
            "high": [11.0 + i for i in range(40)],
            "low": [9.0 + i for i in range(40)],
            "close": [10.0 + i for i in range(40)],
            "volume": [1000.0 + i for i in range(40)],
        },
        index=index,
    )
