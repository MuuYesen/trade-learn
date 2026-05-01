"""TA-Lib indicator namespace backed by the real TA-Lib package."""

from __future__ import annotations

from tradelearn.indicators.base import FunctionIndicator

from .ta_lib_adapter import _atr, _ema, _macd, _rsi, _sma

SMA = FunctionIndicator("talib.SMA", _sma, {"timeperiod": 30})
EMA = FunctionIndicator("talib.EMA", _ema, {"timeperiod": 30})
RSI = FunctionIndicator("talib.RSI", _rsi, {"timeperiod": 14})
MACD = FunctionIndicator(
    "talib.MACD",
    _macd,
    {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
)
ATR = FunctionIndicator("talib.ATR", _atr, {"timeperiod": 14})

sma = SMA
ema = EMA
rsi = RSI
macd = MACD
atr = ATR

__all__ = ["ATR", "EMA", "MACD", "RSI", "SMA", "atr", "ema", "macd", "rsi", "sma"]
