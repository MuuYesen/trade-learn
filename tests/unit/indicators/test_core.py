"""Tests for generic indicator wrappers."""

import pandas as pd
import pandas_ta_classic as pta

from tradelearn import ta


def test_sma_matches_pandas_ta_classic() -> None:
    """ta.sma delegates to pandas-ta-classic SMA."""
    close = _close()

    result = ta.sma(close, period=3)

    pd.testing.assert_series_equal(result, pta.sma(close, length=3))


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
