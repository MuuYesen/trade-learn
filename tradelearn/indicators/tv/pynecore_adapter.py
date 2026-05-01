"""PyneCore-backed TradingView indicator adapters."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pynecore.lib as pyne_lib
import pynecore.lib.ta as pyne_ta
from pynecore.core.function_isolation import isolate_function, reset as reset_pyne_functions
from pynecore.core.series import inline_series
from pynecore.types.na import NA


def _sma(close: pd.Series, length: int = 20) -> pd.Series:
    return _run_source_indicator(close, "sma", lambda source, fn: fn(source, length))


def _ema(close: pd.Series, length: int = 20) -> pd.Series:
    return _run_source_indicator(close, "ema", lambda source, fn: fn(source, length))


def _wma(close: pd.Series, length: int = 20) -> pd.Series:
    return _run_source_indicator(close, "wma", lambda source, fn: fn(source, length))


def _rma(close: pd.Series, length: int = 14) -> pd.Series:
    return _run_source_indicator(close, "rma", lambda source, fn: fn(source, length))


def _hma(close: pd.Series, length: int = 20) -> pd.Series:
    return _run_source_indicator(close, "hma", lambda source, fn: fn(source, length))


def _swma(close: pd.Series) -> pd.Series:
    return _run_source_indicator(close, "swma", lambda source, fn: fn(source))


def _alma(
    close: pd.Series,
    length: int = 9,
    offset: float = 0.85,
    sigma: float = 6.0,
    floor: bool = False,
) -> pd.Series:
    return _run_source_indicator(
        close,
        "alma",
        lambda source, fn: fn(source, length, offset, sigma, floor),
    )


def _stdev(close: pd.Series, length: int = 20, biased: bool = True) -> pd.Series:
    return _run_source_indicator(close, "stdev", lambda source, fn: fn(source, length, biased))


def _variance(close: pd.Series, length: int = 20, biased: bool = True) -> pd.Series:
    return _run_source_indicator(close, "variance", lambda source, fn: fn(source, length, biased))


def _roc(close: pd.Series, length: int = 10) -> pd.Series:
    return _run_source_indicator(close, "roc", lambda source, fn: fn(source, length))


def _mom(close: pd.Series, length: int = 10) -> pd.Series:
    return _run_source_indicator(close, "mom", lambda source, fn: fn(source, length))


def _cmo(close: pd.Series, length: int = 14) -> pd.Series:
    return _run_source_indicator(close, "cmo", lambda source, fn: fn(source, length))


def _tsi(close: pd.Series, short_length: int = 13, long_length: int = 25) -> pd.Series:
    return _run_source_indicator(
        close,
        "tsi",
        lambda source, fn: fn(source, short_length, long_length),
    )


def _change(close: pd.Series, length: int = 1) -> pd.Series:
    return _run_source_indicator(close, "change", lambda source, fn: fn(source, length))


def _cum(close: pd.Series) -> pd.Series:
    return _run_source_indicator(close, "cum", lambda source, fn: fn(source))


def _linreg(close: pd.Series, length: int = 14, offset: int = 0) -> pd.Series:
    return _run_source_indicator(close, "linreg", lambda source, fn: fn(source, length, offset))


def _bbands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    frame = _run_source_frame(
        close,
        "bb",
        lambda source, fn: fn(source, length, std),
        columns=("mid", "upper", "lower"),
    )
    return frame[["lower", "mid", "upper"]]


def _bb(close: pd.Series, length: int = 20, mult: float = 2.0) -> pd.DataFrame:
    return _run_source_frame(
        close,
        "bb",
        lambda source, fn: fn(source, length, mult),
        columns=("mid", "upper", "lower"),
    )


def _bbw(close: pd.Series, length: int = 20, mult: float = 2.0) -> pd.Series:
    return _run_source_indicator(close, "bbw", lambda source, fn: fn(source, length, mult))


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    return _run_source_indicator(close, "rsi", lambda source, fn: fn(source, length))


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    return _run_source_frame(
        close,
        "macd",
        lambda source, fn: fn(source, fast, slow, signal),
        columns=("macd", "signal", "hist"),
    )


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    return _run_ohlcv_indicator(high, low, close, None, "atr", lambda fn: fn(length))


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.DataFrame:
    frame = _run_ohlcv_frame(
        high,
        low,
        close,
        None,
        "dmi",
        lambda fn: fn(length, length),
        columns=("dmp", "dmn", "adx"),
    )
    return frame[["adx", "dmp", "dmn"]]


def _dmi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    smoothing: int = 14,
) -> pd.DataFrame:
    return _run_ohlcv_frame(
        high,
        low,
        close,
        None,
        "dmi",
        lambda fn: fn(length, smoothing),
        columns=("dmp", "dmn", "adx"),
    )


def _tr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    handle_na: bool = False,
) -> pd.Series:
    return _run_ohlcv_indicator(high, low, close, None, "tr", lambda fn: fn(handle_na))


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    close_series = pd.Series(close)
    return _run_ohlcv_indicator(
        close_series,
        close_series,
        close_series,
        volume,
        "obv",
        lambda fn: fn(),
    )


def _sar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    start: float = 0.02,
    inc: float = 0.02,
    max: float = 0.2,
) -> pd.Series:
    return _run_ohlcv_indicator(high, low, close, None, "sar", lambda fn: fn(start, inc, max))


def _stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    return _run_ohlcv_indicator(
        high,
        low,
        close,
        None,
        "stoch",
        lambda fn: fn(pyne_lib.close, pyne_lib.high, pyne_lib.low, length),
    )


def _kc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    mult: float = 2.0,
    use_true_range: bool = True,
) -> pd.DataFrame:
    return _run_ohlcv_frame(
        high,
        low,
        close,
        None,
        "kc",
        lambda fn: fn(pyne_lib.close, length, mult, use_true_range),
        columns=("mid", "upper", "lower"),
    )


def _kcw(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    mult: float = 2.0,
    use_true_range: bool = True,
) -> pd.Series:
    return _run_ohlcv_indicator(
        high,
        low,
        close,
        None,
        "kcw",
        lambda fn: fn(pyne_lib.close, length, mult, use_true_range),
    )


def _cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
) -> pd.Series:
    return _run_ohlcv_indicator(
        high,
        low,
        close,
        None,
        "cci",
        lambda fn: fn(pyne_lib.hlc3, length),
    )


def _mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = 14,
) -> pd.Series:
    return _run_ohlcv_indicator(
        high,
        low,
        close,
        volume,
        "mfi",
        lambda fn: fn(pyne_lib.hlc3, length),
    )


def _vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    return _run_ohlcv_indicator(
        high,
        low,
        close,
        volume,
        "vwap",
        lambda fn: fn(pyne_lib.close, anchor=pyne_lib.bar_index == 0),
    )


def _supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    frame = _run_ohlcv_frame(
        high,
        low,
        close,
        None,
        "supertrend",
        lambda fn: fn(multiplier, length),
        columns=("supertrend", "direction"),
    )
    direction = frame["direction"]
    frame["long"] = frame["supertrend"].where(direction < 0)
    frame["short"] = frame["supertrend"].where(direction > 0)
    return frame[["supertrend", "direction", "long", "short"]]


def _ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    senkou: int = 52,
) -> pd.DataFrame:
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    tenkan_line = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_line = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    span_a = ((tenkan_line + kijun_line) / 2).shift(kijun)
    span_b = ((high.rolling(senkou).max() + low.rolling(senkou).min()) / 2).shift(kijun)
    chikou = close.shift(-kijun)
    return pd.DataFrame(
        {
            "span_a": span_a,
            "span_b": span_b,
            "tenkan": tenkan_line,
            "kijun": kijun_line,
            "chikou": chikou,
        },
        index=close.index,
    )


def _run_source_indicator(
    source: pd.Series,
    name: str,
    call: Callable[[object, Callable], object],
) -> pd.Series:
    values = _run_source_values(source, name, call)
    return pd.Series(values, index=pd.Series(source).index, name=getattr(source, "name", None))


def _run_source_frame(
    source: pd.Series,
    name: str,
    call: Callable[[object, Callable], object],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    rows = _run_source_values(source, name, call)
    return pd.DataFrame(rows, index=pd.Series(source).index, columns=list(columns))


def _run_source_values(
    source: pd.Series,
    name: str,
    call: Callable[[object, Callable], object],
) -> list[object]:
    series = pd.Series(source)
    reset_pyne_functions()
    fn = isolate_function(getattr(pyne_ta, name), name, f"tradelearn.tv.{name}")
    values: list[object] = []
    for index, value in enumerate(series):
        pyne_lib.bar_index = index
        source_value = _pyne_value(value)
        source_series = inline_series(source_value, 0)
        values.append(_to_pandas_value(call(source_series, fn)))
    return values


def _run_ohlcv_indicator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series | None,
    name: str,
    call: Callable[[Callable], object],
) -> pd.Series:
    values = _run_ohlcv_values(high, low, close, volume, name, call)
    return pd.Series(values, index=pd.Series(close).index, name=name)


def _run_ohlcv_frame(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series | None,
    name: str,
    call: Callable[[Callable], object],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    rows = _run_ohlcv_values(high, low, close, volume, name, call)
    return pd.DataFrame(rows, index=pd.Series(close).index, columns=list(columns))


def _run_ohlcv_values(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series | None,
    name: str,
    call: Callable[[Callable], object],
) -> list[object]:
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    volume_series = (
        pd.Series(1.0, index=close_series.index)
        if volume is None
        else pd.Series(volume).reindex(close_series.index)
    )
    reset_pyne_functions()
    fn = isolate_function(getattr(pyne_ta, name), name, f"tradelearn.tv.{name}")
    values: list[object] = []
    for index, (high_value, low_value, close_value, volume_value) in enumerate(
        zip(high_series, low_series, close_series, volume_series, strict=False)
    ):
        _set_pyne_ohlcv(index, high_value, low_value, close_value, volume_value)
        values.append(_to_pandas_value(call(fn)))
    return values


def _set_pyne_ohlcv(
    index: int,
    high: object,
    low: object,
    close: object,
    volume: object,
) -> None:
    high_value = _pyne_value(high)
    low_value = _pyne_value(low)
    close_value = _pyne_value(close)
    volume_value = _pyne_value(volume)
    pyne_lib.bar_index = index
    pyne_lib.high = high_value
    pyne_lib.low = low_value
    pyne_lib.close = close_value
    pyne_lib.volume = volume_value
    pyne_lib.hl2 = (
        NA(float)
        if isinstance(high_value, NA) or isinstance(low_value, NA)
        else (high_value + low_value) / 2
    )
    pyne_lib.hlc3 = (
        NA(float)
        if isinstance(high_value, NA) or isinstance(low_value, NA) or isinstance(close_value, NA)
        else (high_value + low_value + close_value) / 3
    )
    pyne_lib.ohlc4 = (
        NA(float)
        if isinstance(high_value, NA) or isinstance(low_value, NA) or isinstance(close_value, NA)
        else (high_value + low_value + close_value + close_value) / 4
    )


def _pyne_value(value: object) -> float | NA:
    if pd.isna(value):
        return NA(float)
    return float(value)


def _to_pandas_value(value: object) -> object:
    if isinstance(value, NA):
        return None
    if isinstance(value, tuple):
        return tuple(_to_pandas_value(item) for item in value)
    return value


__all__ = [
    "_adx",
    "_alma",
    "_atr",
    "_bb",
    "_bbands",
    "_bbw",
    "_cci",
    "_change",
    "_cmo",
    "_cum",
    "_dmi",
    "_ema",
    "_hma",
    "_ichimoku",
    "_kc",
    "_kcw",
    "_linreg",
    "_macd",
    "_mfi",
    "_mom",
    "_obv",
    "_rma",
    "_roc",
    "_rsi",
    "_run_source_indicator",
    "_sar",
    "_sma",
    "_stdev",
    "_stoch",
    "_supertrend",
    "_swma",
    "_tr",
    "_tsi",
    "_variance",
    "_vwap",
    "_wma",
]
