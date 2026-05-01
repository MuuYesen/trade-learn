"""Tests for TradingView-style indicator namespace bootstrap."""

import pandas as pd
import pynecore.lib.ta as pyne_ta

from tradelearn import ta


def test_tv_sma_rsi_macd_are_batch_callable() -> None:
    close = pd.Series(
        [10.0, 11.0, 10.5, 12.0, 13.0, 12.5, 14.0, 15.0],
        name="close",
    )

    assert ta.tv.sma._func.__module__ == "tradelearn.indicators.tv.pynecore_adapter"
    assert ta.tv.PYNE_TA is pyne_ta

    sma = ta.tv.sma(close, length=3)
    rsi = ta.tv.rsi(close, length=4)

    assert sma.isna().iloc[:2].all()
    assert sma.iloc[2] == 10.5
    assert sma.iloc[3] == 11.166666666666666
    assert rsi.isna().iloc[:4].all()
    assert rsi.notna().any()

    macd = ta.tv.macd(close, fast=3, slow=6, signal=2)

    assert list(macd.columns) == ["macd", "signal", "hist"]
    assert len(macd) == len(close)


def test_tv_exposes_common_pynecore_source_indicators() -> None:
    close = pd.Series(
        [10.0, 11.0, 10.5, 12.0, 13.0, 12.5, 14.0, 15.0],
        name="close",
    )

    for name in [
        "alma",
        "bbw",
        "change",
        "cmo",
        "cum",
        "ema",
        "hma",
        "linreg",
        "mom",
        "roc",
        "rma",
        "stdev",
        "swma",
        "tsi",
        "variance",
        "wma",
    ]:
        indicator = getattr(ta.tv, name)
        result = indicator(close)
        assert len(result) == len(close), name

    bb = ta.tv.bb(close, length=5, mult=2.0)
    assert list(bb.columns) == ["mid", "upper", "lower"]
    assert len(bb) == len(close)


def test_tv_common_ohlcv_indicators_are_batch_callable() -> None:
    frame = _ohlcv()

    bands = ta.tv.bbands(frame.close, length=5, std=2.0)
    assert list(bands.columns) == ["lower", "mid", "upper"]
    assert len(bands) == len(frame)

    adx = ta.tv.adx(frame.high, frame.low, frame.close, length=5)
    assert list(adx.columns) == ["adx", "dmp", "dmn"]
    assert len(adx) == len(frame)

    dmi = ta.tv.dmi(frame.high, frame.low, frame.close, length=5, smoothing=5)
    assert list(dmi.columns) == ["dmp", "dmn", "adx"]
    assert len(dmi) == len(frame)

    tr = ta.tv.tr(frame.high, frame.low, frame.close)
    assert len(tr) == len(frame)

    atr = ta.tv.atr(frame.high, frame.low, frame.close, length=5)
    assert atr.isna().iloc[:4].all()
    assert atr.notna().any()

    obv = ta.tv.obv(frame.close, frame.volume)
    assert len(obv) == len(frame)

    sar = ta.tv.sar(frame.high, frame.low, frame.close)
    assert len(sar) == len(frame)

    stoch = ta.tv.stoch(frame.high, frame.low, frame.close, length=5)
    assert len(stoch) == len(frame)

    kc = ta.tv.kc(frame.high, frame.low, frame.close, length=5, mult=1.5)
    assert list(kc.columns) == ["mid", "upper", "lower"]
    assert len(kc) == len(frame)

    kcw = ta.tv.kcw(frame.high, frame.low, frame.close, length=5, mult=1.5)
    assert len(kcw) == len(frame)

    cci = ta.tv.cci(frame.high, frame.low, frame.close, length=5)
    assert len(cci) == len(frame)

    mfi = ta.tv.mfi(frame.high, frame.low, frame.close, frame.volume, length=5)
    assert len(mfi) == len(frame)

    vwap = ta.tv.vwap(frame.high, frame.low, frame.close, frame.volume)
    assert vwap.notna().all()
    assert vwap.iloc[0] == frame.close.iloc[0]

    supertrend = ta.tv.supertrend(frame.high, frame.low, frame.close, length=7, multiplier=3.0)
    assert list(supertrend.columns) == ["supertrend", "direction", "long", "short"]
    assert len(supertrend) == len(frame)

    ichimoku = ta.tv.ichimoku(frame.high, frame.low, frame.close)
    assert list(ichimoku.columns) == ["span_a", "span_b", "tenkan", "kijun", "chikou"]
    assert len(ichimoku) == len(frame)


def test_tv_namespace_exposes_pynecore_backend_marker() -> None:
    assert ta.tv.BACKEND == "pynecore"
    assert len(ta.tv.COVERED_PYNECORE) >= 30
    assert "ichimoku" in ta.tv.FALLBACKS


def test_tv_namespace_keeps_init_as_facade() -> None:
    assert ta.tv._run_source_indicator.__module__ == "tradelearn.indicators.tv.pynecore_adapter"
    assert ta.tv._sma.__module__ == "tradelearn.indicators.tv.pynecore_adapter"
    assert ta.tv._atr.__module__ == "tradelearn.indicators.tv.pynecore_adapter"


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
