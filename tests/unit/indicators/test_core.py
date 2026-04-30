"""Tests for generic indicator wrappers."""

from typing import Any

import pandas as pd
import pandas_ta_classic as pta
import pytest

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
    """Function-style indicators without bar_columns still raise NotImplementedError."""
    from tradelearn.indicators.base import FunctionIndicator

    def _dummy(close: Any) -> Any:
        return close

    no_stream = FunctionIndicator("dummy", _dummy)
    try:
        no_stream.on_bar(object())
    except NotImplementedError as exc:
        assert "streaming" in str(exc)
    else:
        raise AssertionError("on_bar should raise when bar_columns is not set")


def test_sma_on_bar_streaming() -> None:
    """ta.sma.on_bar accumulates a buffer and returns the current SMA value."""
    from tradelearn.indicators.base import FunctionIndicator

    sma_stream = FunctionIndicator("sma", ta.sma._func, {"period": 3}, bar_columns=("close",))

    class _Bar:
        def __init__(self, close: float) -> None:
            self.close = close

    prices = [10.0, 11.0, 12.0, 13.0]
    results = [sma_stream.on_bar(_Bar(p)) for p in prices]

    # After 3 bars: SMA(3) of [10, 11, 12] = 11.0
    assert results[2] == pytest.approx(11.0)
    # After 4 bars: SMA(3) of [10, 11, 12, 13] → last value = (11+12+13)/3 = 12.0
    assert results[3] == pytest.approx(12.0)
    sma_stream.reset()
    assert sma_stream._bar_buffers is None


def test_atr_on_bar_streaming() -> None:
    """ta.atr.on_bar works with multi-column StreamBar-style objects."""
    from dataclasses import dataclass

    from tradelearn.indicators.base import FunctionIndicator

    atr_stream = FunctionIndicator(
        "atr", ta.atr._func, {"length": 2}, bar_columns=("high", "low", "close")
    )

    @dataclass
    class _Bar:
        high: float
        low: float
        close: float

    bars = [_Bar(11, 9, 10), _Bar(12, 10, 11), _Bar(13, 11, 12)]
    results = [atr_stream.on_bar(b) for b in bars]
    assert results[-1] == results[-1]  # not NaN after warmup


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
