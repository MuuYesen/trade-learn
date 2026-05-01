"""TA-Lib backed indicator adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import talib as _talib
except ImportError as exc:  # pragma: no cover - depends on optional native wheel availability.
    raise ImportError(
        "tradelearn.indicators.talib requires TA-Lib. "
        "Install the TA-Lib Python package to use tl.talib or bt.talib."
    ) from exc


def _series(values: np.ndarray, index: pd.Index | None, name: str) -> pd.Series:
    return pd.Series(values, index=index, name=name)


def _array(values: pd.Series) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _sma(close: pd.Series, timeperiod: int = 30) -> pd.Series:
    return _series(_talib.SMA(_array(close), timeperiod=timeperiod), close.index, "SMA")


def _ema(close: pd.Series, timeperiod: int = 30) -> pd.Series:
    return _series(_talib.EMA(_array(close), timeperiod=timeperiod), close.index, "EMA")


def _rsi(close: pd.Series, timeperiod: int = 14) -> pd.Series:
    return _series(_talib.RSI(_array(close), timeperiod=timeperiod), close.index, "RSI")


def _macd(
    close: pd.Series,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
) -> pd.DataFrame:
    macd, signal, hist = _talib.MACD(
        _array(close),
        fastperiod=fastperiod,
        slowperiod=slowperiod,
        signalperiod=signalperiod,
    )
    return pd.DataFrame(
        {"macd": macd, "signal": signal, "hist": hist},
        index=close.index,
    )


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    timeperiod: int = 14,
) -> pd.Series:
    values = _talib.ATR(
        _array(high),
        _array(low),
        _array(close),
        timeperiod=timeperiod,
    )
    return _series(values, close.index, "ATR")


__all__ = ["_atr", "_ema", "_macd", "_rsi", "_sma"]
