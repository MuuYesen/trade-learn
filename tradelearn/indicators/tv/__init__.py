"""TradingView-style indicator namespace.

This namespace exposes TradingView/Pine-style indicators backed by PyneCore.
PyneCore evaluates indicators bar by bar; :mod:`pynecore_adapter` adapts pandas
inputs into PyneCore and this module keeps the public facade small.
"""

from __future__ import annotations

import pynecore.lib.ta as pyne_ta

from tradelearn.indicators.base import FunctionIndicator

from .pynecore_adapter import (
    _adx,
    _alma,
    _atr,
    _bb,
    _bbands,
    _bbw,
    _cci,
    _change,
    _cmo,
    _cum,
    _dmi,
    _ema,
    _hma,
    _ichimoku,
    _kc,
    _kcw,
    _linreg,
    _macd,
    _mfi,
    _mom,
    _obv,
    _rma,
    _roc,
    _rsi,
    _run_source_indicator,
    _sar,
    _sma,
    _stdev,
    _stoch,
    _supertrend,
    _swma,
    _tr,
    _tsi,
    _variance,
    _vwap,
    _wma,
)

BACKEND = "pynecore"
PYNE_TA = pyne_ta
FALLBACKS = {
    "ichimoku": "pynecore.lib.ta does not expose ichimoku; implemented locally.",
}
COVERED_PYNECORE = {
    "alma",
    "atr",
    "bb",
    "bbw",
    "cci",
    "change",
    "cmo",
    "cum",
    "dmi",
    "ema",
    "hma",
    "kc",
    "kcw",
    "linreg",
    "macd",
    "mfi",
    "mom",
    "obv",
    "rma",
    "roc",
    "rsi",
    "sar",
    "sma",
    "stdev",
    "stoch",
    "supertrend",
    "swma",
    "tr",
    "tsi",
    "variance",
    "vwap",
    "wma",
}

sma = FunctionIndicator("tv.sma", _sma, {"length": 20})
ema = FunctionIndicator("tv.ema", _ema, {"length": 20})
wma = FunctionIndicator("tv.wma", _wma, {"length": 20})
rma = FunctionIndicator("tv.rma", _rma, {"length": 14})
hma = FunctionIndicator("tv.hma", _hma, {"length": 20})
swma = FunctionIndicator("tv.swma", _swma, {})
alma = FunctionIndicator(
    "tv.alma",
    _alma,
    {"length": 9, "offset": 0.85, "sigma": 6.0, "floor": False},
)
stdev = FunctionIndicator("tv.stdev", _stdev, {"length": 20, "biased": True})
variance = FunctionIndicator("tv.variance", _variance, {"length": 20, "biased": True})
roc = FunctionIndicator("tv.roc", _roc, {"length": 10})
mom = FunctionIndicator("tv.mom", _mom, {"length": 10})
cmo = FunctionIndicator("tv.cmo", _cmo, {"length": 14})
tsi = FunctionIndicator("tv.tsi", _tsi, {"short_length": 13, "long_length": 25})
change = FunctionIndicator("tv.change", _change, {"length": 1})
cum = FunctionIndicator("tv.cum", _cum, {})
linreg = FunctionIndicator("tv.linreg", _linreg, {"length": 14, "offset": 0})
bb = FunctionIndicator("tv.bb", _bb, {"length": 20, "mult": 2.0})
bbands = FunctionIndicator("tv.bbands", _bbands, {"length": 20, "std": 2.0})
bbw = FunctionIndicator("tv.bbw", _bbw, {"length": 20, "mult": 2.0})
rsi = FunctionIndicator("tv.rsi", _rsi, {"length": 14})
macd = FunctionIndicator("tv.macd", _macd, {"fast": 12, "slow": 26, "signal": 9})
atr = FunctionIndicator("tv.atr", _atr, {"length": 14})
adx = FunctionIndicator("tv.adx", _adx, {"length": 14})
dmi = FunctionIndicator("tv.dmi", _dmi, {"length": 14, "smoothing": 14})
tr = FunctionIndicator("tv.tr", _tr, {"handle_na": False})
obv = FunctionIndicator("tv.obv", _obv, {})
sar = FunctionIndicator("tv.sar", _sar, {"start": 0.02, "inc": 0.02, "max": 0.2})
stoch = FunctionIndicator("tv.stoch", _stoch, {"length": 14})
kc = FunctionIndicator("tv.kc", _kc, {"length": 20, "mult": 2.0, "use_true_range": True})
kcw = FunctionIndicator("tv.kcw", _kcw, {"length": 20, "mult": 2.0, "use_true_range": True})
cci = FunctionIndicator("tv.cci", _cci, {"length": 20})
mfi = FunctionIndicator("tv.mfi", _mfi, {"length": 14})
vwap = FunctionIndicator("tv.vwap", _vwap, {})
supertrend = FunctionIndicator("tv.supertrend", _supertrend, {"length": 10, "multiplier": 3.0})
ichimoku = FunctionIndicator("tv.ichimoku", _ichimoku, {"tenkan": 9, "kijun": 26, "senkou": 52})

SMA = sma
EMA = ema
WMA = wma
RMA = rma
HMA = hma
SWMA = swma
ALMA = alma
STDEV = stdev
VARIANCE = variance
ROC = roc
MOM = mom
CMO = cmo
TSI = tsi
CHANGE = change
CUM = cum
LINREG = linreg
BB = bb
BBANDS = bbands
BBW = bbw
RSI = rsi
MACD = macd
ATR = atr
ADX = adx
DMI = dmi
TR = tr
OBV = obv
SAR = sar
STOCH = stoch
KC = kc
KCW = kcw
CCI = cci
MFI = mfi
VWAP = vwap
SUPERTREND = supertrend
ICHIMOKU = ichimoku

__all__ = [
    "ADX",
    "ALMA",
    "ATR",
    "BACKEND",
    "BB",
    "BBANDS",
    "BBW",
    "CCI",
    "CHANGE",
    "CMO",
    "COVERED_PYNECORE",
    "CUM",
    "DMI",
    "EMA",
    "FALLBACKS",
    "HMA",
    "ICHIMOKU",
    "KC",
    "KCW",
    "LINREG",
    "MACD",
    "MFI",
    "MOM",
    "OBV",
    "PYNE_TA",
    "RSI",
    "RMA",
    "ROC",
    "SAR",
    "SMA",
    "STDEV",
    "STOCH",
    "SUPERTREND",
    "SWMA",
    "TR",
    "TSI",
    "VARIANCE",
    "VWAP",
    "WMA",
    "adx",
    "alma",
    "atr",
    "bb",
    "bbands",
    "bbw",
    "cci",
    "change",
    "cmo",
    "cum",
    "dmi",
    "ema",
    "hma",
    "ichimoku",
    "kc",
    "kcw",
    "linreg",
    "macd",
    "mfi",
    "mom",
    "obv",
    "rsi",
    "rma",
    "roc",
    "sar",
    "sma",
    "stdev",
    "stoch",
    "supertrend",
    "swma",
    "tr",
    "tsi",
    "variance",
    "vwap",
    "wma",
]
