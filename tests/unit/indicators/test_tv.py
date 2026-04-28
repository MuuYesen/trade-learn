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


def test_tv_namespace_exposes_pynecore_backend_marker() -> None:
    assert ta.tv.BACKEND == "pynecore"

