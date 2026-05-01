"""pandas-ta-classic indicator namespace."""

from __future__ import annotations

from tradelearn.indicators.base import FunctionIndicator

from .pandas_ta_adapter import _atr, _ema, _macd, _rsi, _sma

SMA = FunctionIndicator("pta.SMA", _sma, {"length": 30})
EMA = FunctionIndicator("pta.EMA", _ema, {"length": 30})
RSI = FunctionIndicator("pta.RSI", _rsi, {"length": 14})
MACD = FunctionIndicator("pta.MACD", _macd, {"fast": 12, "slow": 26, "signal": 9})
ATR = FunctionIndicator("pta.ATR", _atr, {"length": 14})

sma = SMA
ema = EMA
rsi = RSI
macd = MACD
atr = ATR

__all__ = ["ATR", "EMA", "MACD", "RSI", "SMA", "atr", "ema", "macd", "rsi", "sma"]
