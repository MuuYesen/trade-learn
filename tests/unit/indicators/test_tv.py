"""Tests for TradingView-style indicator namespace bootstrap."""

import pandas as pd
import pandas_ta_classic as pta

from tradelearn import ta


def test_tv_sma_rsi_macd_are_batch_callable() -> None:
    close = pd.Series(
        [10.0, 11.0, 10.5, 12.0, 13.0, 12.5, 14.0, 15.0],
        name="close",
    )

    pd.testing.assert_series_equal(ta.tv.sma(close, length=3), pta.sma(close, length=3))
    pd.testing.assert_series_equal(ta.tv.rsi(close, length=4), pta.rsi(close, length=4))

    macd = ta.tv.macd(close, fast=3, slow=6, signal=2)

    assert list(macd.columns) == ["macd", "signal", "hist"]
    assert len(macd) == len(close)


def test_tv_common_ohlcv_indicators_are_batch_callable() -> None:
    frame = _ohlcv()

    bands = ta.tv.bbands(frame.close, length=5, std=2.0)
    assert list(bands.columns) == ["lower", "mid", "upper"]
    assert len(bands) == len(frame)

    adx = ta.tv.adx(frame.high, frame.low, frame.close, length=5)
    assert list(adx.columns) == ["adx", "dmp", "dmn"]
    assert len(adx) == len(frame)

    pd.testing.assert_series_equal(
        ta.tv.atr(frame.high, frame.low, frame.close, length=5),
        pta.atr(frame.high, frame.low, frame.close, length=5),
    )
    pd.testing.assert_series_equal(
        ta.tv.vwap(frame.high, frame.low, frame.close, frame.volume),
        pta.vwap(frame.high, frame.low, frame.close, frame.volume),
    )

    supertrend = ta.tv.supertrend(frame.high, frame.low, frame.close, length=7, multiplier=3.0)
    assert list(supertrend.columns) == ["supertrend", "direction", "long", "short"]
    assert len(supertrend) == len(frame)

    ichimoku = ta.tv.ichimoku(frame.high, frame.low, frame.close)
    assert list(ichimoku.columns) == ["span_a", "span_b", "tenkan", "kijun", "chikou"]
    assert len(ichimoku) == len(frame)


def test_tv_namespace_exposes_pynecore_backend_marker() -> None:
    assert ta.tv.BACKEND == "pynecore"


def _ohlcv() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=90, freq="D")
    return pd.DataFrame(
        {
            "high": [11.0 + i for i in range(90)],
            "low": [9.0 + i for i in range(90)],
            "close": [10.0 + i for i in range(90)],
            "volume": [1000.0 + i for i in range(90)],
        },
        index=index,
    )
