"""Tongdaxin 30 classic indicator wrappers backed by MyTT."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from tradelearn.indicators.base import FunctionIndicator
from tradelearn.query.tec import MyTT


def _index(values: pd.Series | Sequence[float]) -> pd.Index | None:
    """Return the index for pandas inputs."""
    if isinstance(values, pd.Series):
        return values.index
    return None


def _array(values: pd.Series | Sequence[float]) -> Sequence[float]:
    """Return raw values for MyTT functions."""
    if isinstance(values, pd.Series):
        return values.to_numpy()
    return values


def _series(
    values: Sequence[float],
    *,
    index: pd.Index | None,
    name: str,
) -> pd.Series:
    """Create a named series with an optional source index."""
    return pd.Series(values, index=index, name=name)


def _frame(
    columns: dict[str, Sequence[float]],
    *,
    index: pd.Index | None,
) -> pd.DataFrame:
    """Create a frame from named MyTT output arrays."""
    return pd.DataFrame(columns, index=index)


def _unexpected(kwargs: dict[str, int | float]) -> None:
    """Raise for unsupported keyword arguments."""
    if kwargs:
        raise TypeError(f"Unexpected keyword argument: {next(iter(kwargs))}")


def _ma(close: pd.Series | Sequence[float], n: int = 5, **kwargs: int) -> pd.Series:
    """Simple moving average using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    _unexpected(kwargs)
    return _series(MyTT.MA(_array(close), n), index=_index(close), name=f"MA_{n}")


def _ema(close: pd.Series | Sequence[float], n: int = 5, **kwargs: int) -> pd.Series:
    """Exponential moving average using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    _unexpected(kwargs)
    return _series(MyTT.EMA(_array(close), n), index=_index(close), name=f"EMA_{n}")


def _sma(
    close: pd.Series | Sequence[float],
    n: int = 5,
    m: int = 1,
    **kwargs: int,
) -> pd.Series:
    """Chinese-style SMA using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    m = kwargs.pop("M", m)
    _unexpected(kwargs)
    return _series(MyTT.SMA(_array(close), n, m), index=_index(close), name=f"SMA_{n}_{m}")


def _wma(close: pd.Series | Sequence[float], n: int = 5, **kwargs: int) -> pd.Series:
    """Weighted moving average using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    _unexpected(kwargs)
    return _series(MyTT.WMA(_array(close), n), index=_index(close), name=f"WMA_{n}")


def _macd(
    close: pd.Series | Sequence[float],
    short: int = 12,
    long: int = 26,
    m: int = 9,
    **kwargs: int,
) -> pd.DataFrame:
    """MACD using Tongdaxin semantics."""
    short = kwargs.pop("SHORT", short)
    long = kwargs.pop("LONG", long)
    m = kwargs.pop("M", m)
    _unexpected(kwargs)
    dif, dea, macd_value = MyTT.MACD(_array(close), SHORT=short, LONG=long, M=m)
    return _frame({"DIF": dif, "DEA": dea, "MACD": macd_value}, index=_index(close))


def _kdj(
    close: pd.Series | Sequence[float],
    high: pd.Series | Sequence[float],
    low: pd.Series | Sequence[float],
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
    **kwargs: int,
) -> pd.DataFrame:
    """KDJ using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    m1 = kwargs.pop("M1", m1)
    m2 = kwargs.pop("M2", m2)
    _unexpected(kwargs)
    k, d, j = MyTT.KDJ(
        _array(close),
        _array(high),
        _array(low),
        N=n,
        M1=m1,
        M2=m2,
    )
    return _frame({"K": k, "D": d, "J": j}, index=_index(close))


def _rsi(close: pd.Series | Sequence[float], n: int = 24, **kwargs: int) -> pd.Series:
    """RSI using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    _unexpected(kwargs)
    return _series(MyTT.RSI(_array(close), N=n), index=_index(close), name=f"RSI_{n}")


def _wr(
    close: pd.Series | Sequence[float],
    high: pd.Series | Sequence[float],
    low: pd.Series | Sequence[float],
    n: int = 10,
    n1: int = 6,
    **kwargs: int,
) -> pd.DataFrame:
    """Williams %R using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    n1 = kwargs.pop("N1", n1)
    _unexpected(kwargs)
    wr_value, wr1 = MyTT.WR(_array(close), _array(high), _array(low), N=n, N1=n1)
    return _frame({"WR": wr_value, "WR1": wr1}, index=_index(close))


def _bias(
    close: pd.Series | Sequence[float],
    l1: int = 6,
    l2: int = 12,
    l3: int = 24,
    **kwargs: int,
) -> pd.DataFrame:
    """BIAS using Tongdaxin semantics."""
    l1 = kwargs.pop("L1", l1)
    l2 = kwargs.pop("L2", l2)
    l3 = kwargs.pop("L3", l3)
    _unexpected(kwargs)
    bias1, bias2, bias3 = MyTT.BIAS(_array(close), L1=l1, L2=l2, L3=l3)
    return _frame({"BIAS1": bias1, "BIAS2": bias2, "BIAS3": bias3}, index=_index(close))


def _boll(
    close: pd.Series | Sequence[float],
    n: int = 20,
    p: float = 2,
    **kwargs: int | float,
) -> pd.DataFrame:
    """Bollinger bands using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    p = kwargs.pop("P", p)
    _unexpected(kwargs)
    upper, mid, lower = MyTT.BOLL(_array(close), N=n, P=p)
    return _frame({"UPPER": upper, "MID": mid, "LOWER": lower}, index=_index(close))


def _psy(
    close: pd.Series | Sequence[float],
    n: int = 12,
    m: int = 6,
    **kwargs: int,
) -> pd.DataFrame:
    """Psychological line using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    m = kwargs.pop("M", m)
    _unexpected(kwargs)
    psy_value, psyma = MyTT.PSY(_array(close), N=n, M=m)
    return _frame({"PSY": psy_value, "PSYMA": psyma}, index=_index(close))


def _cci(
    close: pd.Series | Sequence[float],
    high: pd.Series | Sequence[float],
    low: pd.Series | Sequence[float],
    n: int = 14,
    **kwargs: int,
) -> pd.Series:
    """CCI using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    _unexpected(kwargs)
    values = MyTT.CCI(_array(close), _array(high), _array(low), N=n)
    return _series(values, index=_index(close), name=f"CCI_{n}")


def _atr(
    close: pd.Series | Sequence[float],
    high: pd.Series | Sequence[float],
    low: pd.Series | Sequence[float],
    n: int = 20,
    **kwargs: int,
) -> pd.Series:
    """ATR using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    _unexpected(kwargs)
    values = MyTT.ATR(_array(close), _array(high), _array(low), N=n)
    return _series(values, index=_index(close), name=f"ATR_{n}")


def _bbi(
    close: pd.Series | Sequence[float],
    m1: int = 3,
    m2: int = 6,
    m3: int = 12,
    m4: int = 20,
    **kwargs: int,
) -> pd.Series:
    """BBI using Tongdaxin semantics."""
    m1 = kwargs.pop("M1", m1)
    m2 = kwargs.pop("M2", m2)
    m3 = kwargs.pop("M3", m3)
    m4 = kwargs.pop("M4", m4)
    _unexpected(kwargs)
    values = MyTT.BBI(_array(close), M1=m1, M2=m2, M3=m3, M4=m4)
    return _series(values, index=_index(close), name="BBI")


def _dmi(
    close: pd.Series | Sequence[float],
    high: pd.Series | Sequence[float],
    low: pd.Series | Sequence[float],
    m1: int = 14,
    m2: int = 6,
    **kwargs: int,
) -> pd.DataFrame:
    """DMI using Tongdaxin semantics."""
    m1 = kwargs.pop("M1", m1)
    m2 = kwargs.pop("M2", m2)
    _unexpected(kwargs)
    pdi, mdi, adx, adxr = MyTT.DMI(
        _array(close),
        _array(high),
        _array(low),
        M1=m1,
        M2=m2,
    )
    return _frame({"PDI": pdi, "MDI": mdi, "ADX": adx, "ADXR": adxr}, index=_index(close))


def _trix(
    close: pd.Series | Sequence[float],
    m1: int = 12,
    m2: int = 20,
    **kwargs: int,
) -> pd.DataFrame:
    """TRIX using Tongdaxin semantics."""
    m1 = kwargs.pop("M1", m1)
    m2 = kwargs.pop("M2", m2)
    _unexpected(kwargs)
    trix_value, trma = MyTT.TRIX(_array(close), M1=m1, M2=m2)
    return _frame({"TRIX": trix_value, "TRMA": trma}, index=_index(close))


def _vr(
    close: pd.Series | Sequence[float],
    volume: pd.Series | Sequence[float],
    m1: int = 26,
    **kwargs: int,
) -> pd.Series:
    """VR using Tongdaxin semantics."""
    m1 = kwargs.pop("M1", m1)
    _unexpected(kwargs)
    values = MyTT.VR(_array(close), _array(volume), M1=m1)
    return _series(values, index=_index(close), name=f"VR_{m1}")


def _mtm(
    close: pd.Series | Sequence[float],
    n: int = 12,
    m: int = 6,
    **kwargs: int,
) -> pd.DataFrame:
    """MTM using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    m = kwargs.pop("M", m)
    _unexpected(kwargs)
    mtm_value, mtmma = MyTT.MTM(_array(close), N=n, M=m)
    return _frame({"MTM": mtm_value, "MTMMA": mtmma}, index=_index(close))


def _roc(
    close: pd.Series | Sequence[float],
    n: int = 12,
    m: int = 6,
    **kwargs: int,
) -> pd.DataFrame:
    """ROC using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    m = kwargs.pop("M", m)
    _unexpected(kwargs)
    roc_value, maroc = MyTT.ROC(_array(close), N=n, M=m)
    return _frame({"ROC": roc_value, "MAROC": maroc}, index=_index(close))


def _taq(
    high: pd.Series | Sequence[float],
    low: pd.Series | Sequence[float],
    n: int = 20,
    **kwargs: int,
) -> pd.DataFrame:
    """TAQ channel using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    _unexpected(kwargs)
    up, mid, down = MyTT.TAQ(_array(high), _array(low), N=n)
    return _frame({"UP": up, "MID": mid, "DOWN": down}, index=_index(high))


def _ktn(
    close: pd.Series | Sequence[float],
    high: pd.Series | Sequence[float],
    low: pd.Series | Sequence[float],
    n: int = 20,
    m: int = 10,
    **kwargs: int,
) -> pd.DataFrame:
    """Keltner channel using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    m = kwargs.pop("M", m)
    _unexpected(kwargs)
    upper, mid, lower = MyTT.KTN(_array(close), _array(high), _array(low), N=n, M=m)
    return _frame({"UPPER": upper, "MID": mid, "LOWER": lower}, index=_index(close))


def _cr(
    close: pd.Series | Sequence[float],
    high: pd.Series | Sequence[float],
    low: pd.Series | Sequence[float],
    n: int = 20,
    **kwargs: int,
) -> pd.Series:
    """CR using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    _unexpected(kwargs)
    values = MyTT.CR(_array(close), _array(high), _array(low), N=n)
    return _series(values, index=_index(close), name=f"CR_{n}")


def _emv(
    high: pd.Series | Sequence[float],
    low: pd.Series | Sequence[float],
    volume: pd.Series | Sequence[float],
    n: int = 14,
    m: int = 9,
    **kwargs: int,
) -> pd.DataFrame:
    """EMV using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    m = kwargs.pop("M", m)
    _unexpected(kwargs)
    emv_value, maemv = MyTT.EMV(_array(high), _array(low), _array(volume), N=n, M=m)
    return _frame({"EMV": emv_value, "MAEMV": maemv}, index=_index(high))


def _dpo(
    close: pd.Series | Sequence[float],
    m1: int = 20,
    m2: int = 10,
    m3: int = 6,
    **kwargs: int,
) -> pd.DataFrame:
    """DPO using Tongdaxin semantics."""
    m1 = kwargs.pop("M1", m1)
    m2 = kwargs.pop("M2", m2)
    m3 = kwargs.pop("M3", m3)
    _unexpected(kwargs)
    dpo_value, madpo = MyTT.DPO(_array(close), M1=m1, M2=m2, M3=m3)
    return _frame({"DPO": dpo_value, "MADPO": madpo}, index=_index(close))


def _brar(
    open_: pd.Series | Sequence[float],
    close: pd.Series | Sequence[float],
    high: pd.Series | Sequence[float],
    low: pd.Series | Sequence[float],
    m1: int = 26,
    **kwargs: int,
) -> pd.DataFrame:
    """BRAR using Tongdaxin semantics."""
    m1 = kwargs.pop("M1", m1)
    _unexpected(kwargs)
    ar, br = MyTT.BRAR(_array(open_), _array(close), _array(high), _array(low), M1=m1)
    return _frame({"AR": ar, "BR": br}, index=_index(close))


def _dfma(
    close: pd.Series | Sequence[float],
    n1: int = 10,
    n2: int = 50,
    m: int = 10,
    **kwargs: int,
) -> pd.DataFrame:
    """DFMA using Tongdaxin semantics."""
    n1 = kwargs.pop("N1", n1)
    n2 = kwargs.pop("N2", n2)
    m = kwargs.pop("M", m)
    _unexpected(kwargs)
    dif, difma = MyTT.DFMA(_array(close), N1=n1, N2=n2, M=m)
    return _frame({"DIF": dif, "DIFMA": difma}, index=_index(close))


def _mass(
    high: pd.Series | Sequence[float],
    low: pd.Series | Sequence[float],
    n1: int = 9,
    n2: int = 25,
    m: int = 6,
    **kwargs: int,
) -> pd.DataFrame:
    """MASS using Tongdaxin semantics."""
    n1 = kwargs.pop("N1", n1)
    n2 = kwargs.pop("N2", n2)
    m = kwargs.pop("M", m)
    _unexpected(kwargs)
    mass_value, ma_mass = MyTT.MASS(_array(high), _array(low), N1=n1, N2=n2, M=m)
    return _frame({"MASS": mass_value, "MA_MASS": ma_mass}, index=_index(high))


def _expma(
    close: pd.Series | Sequence[float],
    n1: int = 12,
    n2: int = 50,
    **kwargs: int,
) -> pd.DataFrame:
    """EXPMA using Tongdaxin semantics."""
    n1 = kwargs.pop("N1", n1)
    n2 = kwargs.pop("N2", n2)
    _unexpected(kwargs)
    expma1, expma2 = MyTT.EXPMA(_array(close), N1=n1, N2=n2)
    return _frame({"EXPMA1": expma1, "EXPMA2": expma2}, index=_index(close))


def _obv(
    close: pd.Series | Sequence[float],
    volume: pd.Series | Sequence[float],
) -> pd.Series:
    """OBV using Tongdaxin semantics."""
    values = MyTT.OBV(_array(close), _array(volume))
    return _series(values, index=_index(close), name="OBV")


def _mfi(
    close: pd.Series | Sequence[float],
    high: pd.Series | Sequence[float],
    low: pd.Series | Sequence[float],
    volume: pd.Series | Sequence[float],
    n: int = 14,
    **kwargs: int,
) -> pd.Series:
    """MFI using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    _unexpected(kwargs)
    values = MyTT.MFI(_array(close), _array(high), _array(low), _array(volume), N=n)
    return _series(values, index=_index(close), name=f"MFI_{n}")


def _asi(
    open_: pd.Series | Sequence[float],
    close: pd.Series | Sequence[float],
    high: pd.Series | Sequence[float],
    low: pd.Series | Sequence[float],
    m1: int = 26,
    m2: int = 10,
    **kwargs: int,
) -> pd.DataFrame:
    """ASI using Tongdaxin semantics."""
    m1 = kwargs.pop("M1", m1)
    m2 = kwargs.pop("M2", m2)
    _unexpected(kwargs)
    asi_value, asit = MyTT.ASI(
        _array(open_),
        _array(close),
        _array(high),
        _array(low),
        M1=m1,
        M2=m2,
    )
    return _frame({"ASI": asi_value, "ASIT": asit}, index=_index(close))


def _xsii(
    close: pd.Series | Sequence[float],
    high: pd.Series | Sequence[float],
    low: pd.Series | Sequence[float],
    n: int = 102,
    m: int = 7,
    **kwargs: int,
) -> pd.DataFrame:
    """XSII using Tongdaxin semantics."""
    n = kwargs.pop("N", n)
    m = kwargs.pop("M", m)
    _unexpected(kwargs)
    td1, td2, td3, td4 = MyTT.XSII(_array(close), _array(high), _array(low), N=n, M=m)
    return _frame({"TD1": td1, "TD2": td2, "TD3": td3, "TD4": td4}, index=_index(close))


ma = FunctionIndicator("tdx.ma", _ma, {"n": 5})
MA = FunctionIndicator("tdx.MA", _ma, {"N": 5})
ema = FunctionIndicator("tdx.ema", _ema, {"n": 5})
EMA = FunctionIndicator("tdx.EMA", _ema, {"N": 5})
sma = FunctionIndicator("tdx.sma", _sma, {"n": 5, "m": 1})
SMA = FunctionIndicator("tdx.SMA", _sma, {"N": 5, "M": 1})
wma = FunctionIndicator("tdx.wma", _wma, {"n": 5})
WMA = FunctionIndicator("tdx.WMA", _wma, {"N": 5})
macd = FunctionIndicator("tdx.macd", _macd, {"short": 12, "long": 26, "m": 9})
MACD = FunctionIndicator("tdx.MACD", _macd, {"SHORT": 12, "LONG": 26, "M": 9})
kdj = FunctionIndicator("tdx.kdj", _kdj, {"n": 9, "m1": 3, "m2": 3})
KDJ = FunctionIndicator("tdx.KDJ", _kdj, {"N": 9, "M1": 3, "M2": 3})
rsi = FunctionIndicator("tdx.rsi", _rsi, {"n": 24})
RSI = FunctionIndicator("tdx.RSI", _rsi, {"N": 24})
wr = FunctionIndicator("tdx.wr", _wr, {"n": 10, "n1": 6})
WR = FunctionIndicator("tdx.WR", _wr, {"N": 10, "N1": 6})
bias = FunctionIndicator("tdx.bias", _bias, {"l1": 6, "l2": 12, "l3": 24})
BIAS = FunctionIndicator("tdx.BIAS", _bias, {"L1": 6, "L2": 12, "L3": 24})
boll = FunctionIndicator("tdx.boll", _boll, {"n": 20, "p": 2})
BOLL = FunctionIndicator("tdx.BOLL", _boll, {"N": 20, "P": 2})
psy = FunctionIndicator("tdx.psy", _psy, {"n": 12, "m": 6})
PSY = FunctionIndicator("tdx.PSY", _psy, {"N": 12, "M": 6})
cci = FunctionIndicator("tdx.cci", _cci, {"n": 14})
CCI = FunctionIndicator("tdx.CCI", _cci, {"N": 14})
atr = FunctionIndicator("tdx.atr", _atr, {"n": 20})
ATR = FunctionIndicator("tdx.ATR", _atr, {"N": 20})
bbi = FunctionIndicator("tdx.bbi", _bbi, {"m1": 3, "m2": 6, "m3": 12, "m4": 20})
BBI = FunctionIndicator("tdx.BBI", _bbi, {"M1": 3, "M2": 6, "M3": 12, "M4": 20})
dmi = FunctionIndicator("tdx.dmi", _dmi, {"m1": 14, "m2": 6})
DMI = FunctionIndicator("tdx.DMI", _dmi, {"M1": 14, "M2": 6})
trix = FunctionIndicator("tdx.trix", _trix, {"m1": 12, "m2": 20})
TRIX = FunctionIndicator("tdx.TRIX", _trix, {"M1": 12, "M2": 20})
vr = FunctionIndicator("tdx.vr", _vr, {"m1": 26})
VR = FunctionIndicator("tdx.VR", _vr, {"M1": 26})
mtm = FunctionIndicator("tdx.mtm", _mtm, {"n": 12, "m": 6})
MTM = FunctionIndicator("tdx.MTM", _mtm, {"N": 12, "M": 6})
roc = FunctionIndicator("tdx.roc", _roc, {"n": 12, "m": 6})
ROC = FunctionIndicator("tdx.ROC", _roc, {"N": 12, "M": 6})
taq = FunctionIndicator("tdx.taq", _taq, {"n": 20})
TAQ = FunctionIndicator("tdx.TAQ", _taq, {"N": 20})
ktn = FunctionIndicator("tdx.ktn", _ktn, {"n": 20, "m": 10})
KTN = FunctionIndicator("tdx.KTN", _ktn, {"N": 20, "M": 10})
cr = FunctionIndicator("tdx.cr", _cr, {"n": 20})
CR = FunctionIndicator("tdx.CR", _cr, {"N": 20})
emv = FunctionIndicator("tdx.emv", _emv, {"n": 14, "m": 9})
EMV = FunctionIndicator("tdx.EMV", _emv, {"N": 14, "M": 9})
dpo = FunctionIndicator("tdx.dpo", _dpo, {"m1": 20, "m2": 10, "m3": 6})
DPO = FunctionIndicator("tdx.DPO", _dpo, {"M1": 20, "M2": 10, "M3": 6})
brar = FunctionIndicator("tdx.brar", _brar, {"m1": 26})
BRAR = FunctionIndicator("tdx.BRAR", _brar, {"M1": 26})
dfma = FunctionIndicator("tdx.dfma", _dfma, {"n1": 10, "n2": 50, "m": 10})
DFMA = FunctionIndicator("tdx.DFMA", _dfma, {"N1": 10, "N2": 50, "M": 10})
mass = FunctionIndicator("tdx.mass", _mass, {"n1": 9, "n2": 25, "m": 6})
MASS = FunctionIndicator("tdx.MASS", _mass, {"N1": 9, "N2": 25, "M": 6})
expma = FunctionIndicator("tdx.expma", _expma, {"n1": 12, "n2": 50})
EXPMA = FunctionIndicator("tdx.EXPMA", _expma, {"N1": 12, "N2": 50})
obv = FunctionIndicator("tdx.obv", _obv, {})
OBV = FunctionIndicator("tdx.OBV", _obv, {})
mfi = FunctionIndicator("tdx.mfi", _mfi, {"n": 14})
MFI = FunctionIndicator("tdx.MFI", _mfi, {"N": 14})
asi = FunctionIndicator("tdx.asi", _asi, {"m1": 26, "m2": 10})
ASI = FunctionIndicator("tdx.ASI", _asi, {"M1": 26, "M2": 10})
xsii = FunctionIndicator("tdx.xsii", _xsii, {"n": 102, "m": 7})
XSII = FunctionIndicator("tdx.XSII", _xsii, {"N": 102, "M": 7})

__all__ = [
    "ASI",
    "ATR",
    "BBI",
    "BIAS",
    "BOLL",
    "BRAR",
    "CCI",
    "CR",
    "DFMA",
    "DMI",
    "DPO",
    "EMA",
    "EMV",
    "EXPMA",
    "KDJ",
    "KTN",
    "MA",
    "MACD",
    "MASS",
    "MFI",
    "MTM",
    "OBV",
    "PSY",
    "ROC",
    "RSI",
    "SMA",
    "TAQ",
    "TRIX",
    "VR",
    "WMA",
    "WR",
    "XSII",
    "asi",
    "atr",
    "bbi",
    "bias",
    "boll",
    "brar",
    "cci",
    "cr",
    "dfma",
    "dmi",
    "dpo",
    "ema",
    "emv",
    "expma",
    "kdj",
    "ktn",
    "ma",
    "macd",
    "mass",
    "mfi",
    "mtm",
    "obv",
    "psy",
    "roc",
    "rsi",
    "sma",
    "taq",
    "trix",
    "vr",
    "wma",
    "wr",
    "xsii",
]
